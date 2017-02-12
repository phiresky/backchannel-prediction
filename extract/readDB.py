#!/usr/bin/env python3

# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/readDB.tcl
# (sha1sum 0e05e903997432917d5ef5b2ef4d799e196f3ffc)
# on 2016-11-06
import itertools
from pprint import pprint

from typing import Set, Dict, List, Iterable, Tuple
import logging
import os
from distutils.dir_util import mkpath
import numpy as np
import time
import sys
import json
import subprocess
from collections import OrderedDict, Counter
import os.path
import re
import functools
from .features import Features
from tqdm import tqdm
from . import util
from .feature import Feature
import hashlib
import random

os.environ['JOBLIB_START_METHOD'] = 'forkserver'
from joblib import Parallel, delayed
import pickle

MAX_TIME = 100 * 60 * 60  # 100 hours
noise_isl = ['<SIL>', '<NOISE>', 'LAUGHTER']
noise_orig = ['[silence]', '[laughter]', '[noise]', '[vocalized-noise]']

DBEntry = Dict[str, str]
DBase = Dict[str, DBEntry]


def isl_noise_filter(text):
    words = []
    for word in text.split(" "):
        if word.startswith("LAUGHTER-"):
            word = word[len("LAUGHTER-"):]
        if word not in noise_isl:
            words.append(word)

    return " ".join(words)


@functools.lru_cache()
def orig_noise_filter(text: str):
    words = []
    for word in text.split(" "):
        if word.startswith("[laughter-"):  # e.g. [laughter-um-hum]
            word = word[len("[laughter-"):-1]
        if word.endswith("_1"):  # e.g. okay_1
            word = word[:-2]
        if word not in noise_orig:
            words.append(word)
    return " ".join(words)


def fromiter(iterator, dtype, shape):
    """Generalises `numpy.fromiter()` to multi-dimesional arrays.

    Instead of the number of elements, the parameter `shape` has to be given,
    which contains the shape of the output array. The first dimension may be
    `-1`, in which case it is inferred from the iterator.
    """
    res_shape = shape[1:]
    if not res_shape:  # Fallback to the "normal" fromiter in the 1-D case
        return np.fromiter(iterator, dtype, shape[0])

    # This wrapping of the iterator is necessary because when used with the
    # field trick, np.fromiter does not enforce consistency of the shapes
    # returned with the '_' field and silently cuts additional elements.
    def shape_checker(iterator, res_shape):
        for value in iterator:
            if value.shape != res_shape:
                raise ValueError("shape of returned object %s does not match"
                                 " given shape %s" % (value.shape, res_shape))
            yield value,

    return np.fromiter(shape_checker(iterator, res_shape),
                       [("_", dtype, res_shape)], shape[0])["_"]


def speakerFilter(convIDs: Set[str], speaker: str) -> bool:
    shortID = speaker.split("-")[0]
    return shortID in convIDs


class DBReader:
    backchannels = None

    noise_filter = None
    uttDB = None
    method = None

    def __init__(self, config_path: str, originalDb=None):
        config = util.load_config(config_path)
        self.config = config
        self.extract_config = self.config['extract_config']
        self.paths_config = self.config['paths']
        self.use_original_db = originalDb if originalDb is not None else self.extract_config['useOriginalDB']
        self.features = Features(config, config_path)
        self.load_db()
        self.backchannels, self.categories = load_backchannels(self.paths_config['backchannels'])
        self.category_to_index = {category: (index + 1) for index, category in enumerate(self.categories)}
        self.index_to_category = {(index + 1): category for index, category in enumerate(self.categories)}
        self.bc_to_category = {bc: category for category, bcs in self.categories.items() for bc in bcs}
        self.sample_window_ms = self.extract_config['sample_window_ms']
        self.method = self.extract_config['extraction_method']

    def is_backchannel(self, uttInfo: dict, index: int, utts: List[Tuple[str, DBEntry]]):
        uttText = uttInfo['text']
        uttText = self.noise_filter(uttText)
        if index == 0:
            # first utterance can't be a backchannel
            return False
        fromTime = float(uttInfo['from'])
        if fromTime + self.method['nbc'][0] < 0:
            return False
        lastUttText = utts[index - 1][1]['text']
        lastUttText = self.noise_filter(lastUttText)
        return (uttText.lower() in self.backchannels and
                (len(lastUttText) == 0 or self.is_backchannel(utts[index - 1][1], index - 1, utts)))

    def get_gauss_bc_feature(self, convid: str):
        return self.get_gauss_bcs(self.features.get_power(convid), convid)

    def getBcRealStartTime(self, utt: str):
        if self.use_original_db:
            for word in self.uttDB.get_words_for_utterance(utt, self.uttDB[utt]):
                text = self.noise_filter(word['text'])
                if len(text) > 0:
                    return float(word['from'])
        else:
            raise Exception("use original db")
            # return self.getBCMaxTime(utt) - 0.3

    def getBcRealFirstEndTime(self, utt: str):
        if self.use_original_db:
            for word in self.uttDB.get_words_for_utterance(utt, self.uttDB[utt]):
                text = self.noise_filter(word['text'])
                if len(text) > 0:
                    return float(word['to'])
        else:
            raise Exception("use original db")
            # return self.getBCMaxTime(utt) - 0.3

    def get_max_time(self, feat: Feature, fromTime: float, toTime: float):
        powerrange = feat.cut_by_time(fromTime, toTime)
        maxTime = fromTime + powerrange.sample_index_to_time(powerrange.argmax())
        return maxTime

    @functools.lru_cache(maxsize=16)
    def getBCMaxTime(self, utt: str):
        uttInfo = self.uttDB[utt]
        uttFrom = float(uttInfo['from'])
        uttTo = float(uttInfo['to'])
        return self.get_max_time(self.features.get_power(uttInfo['convid']), uttFrom, uttTo)

    def getBackchannelTrainingRange(self, utt: str):
        bc_start_time = self.getBcRealStartTime(utt)
        start, end = self.method['bc']
        return bc_start_time + start, bc_start_time + end

    def getNonBackchannelTrainingRange(self, utt: str):
        bc_start_time = self.getBcRealStartTime(utt)
        start, end = self.method['nbc']
        return bc_start_time + start, bc_start_time + end

    def load_db(self) -> (DBase, DBase):
        if self.use_original_db:
            uttDB = UttDB(self.config)
            self.noise_filter = orig_noise_filter
            self.uttDB = uttDB
        else:
            raise Exception("ISL db not supported anymore")

    def get_utterances(self, spkr: str) -> Iterable[Tuple[str, DBEntry]]:
        return self.uttDB.get_utterances(spkr)

    def get_backchannels(self, utts: List[Tuple[str, DBEntry]]) -> List[Tuple[str, DBEntry]]:
        return [(utt, uttInfo)
                for index, (utt, uttInfo) in enumerate(utts)
                if self.is_backchannel(uttInfo, index, utts)
                ]

    def get_gauss_bc_array(self, power: Feature, utt: str) -> Tuple[float, np.array]:
        middle = self.getBcRealStartTime(utt) + self.method['peak']
        stddev = self.method['stddev']
        radius = self.method['radius']
        times = np.arange(-radius, radius, power.frame_shift_ms / 1000, dtype="float32")
        mean = 0
        variance = stddev ** 2
        return middle - radius, (1 / np.sqrt(2 * variance * np.pi)) * np.exp(-((times - mean) ** 2) / (2 * variance))

    def get_gauss_bcs(self, power_feature: np.array, spkr: str):
        arr = np.zeros_like(power_feature)
        feat = Feature(arr, infofrom=power_feature)
        for bc, _ in self.get_backchannels(list(self.get_utterances(spkr))):
            offset, gauss = self.get_gauss_bc_array(power_feature, bc)
            gauss = gauss.reshape((gauss.shape[0], 1))
            start = feat.time_to_sample_index(offset)
            if start < 0:
                gauss = gauss[0 - start:]
                start = 0
            if len(gauss) > arr.size - start:
                gauss = gauss[:arr.size - start]
            feat[start:start + len(gauss)] += gauss
        return feat


@functools.lru_cache(1)
def load_backchannels(path):
    categories = {}
    catnames = {}
    bcs = set()
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            line = line.split(":")
            if len(line) == 2:
                category = line[0]
                category = catnames.setdefault(category[0], category)
                line = line[1]
            elif len(line) == 1:
                category = "unknown"
                line = line[0]
            else:
                raise Exception(f"could not parse line {line}")
            bcs.add(line)
            categoryset = categories.setdefault(category, set())
            categoryset.add(line)
    return set(bcs), categories


def swap_speaker(convid: str):
    (conv, speaker) = convid.split("-")
    other_channel = dict(A="B", B="A")[speaker]
    return conv + "-" + other_channel


# todo: adjust to new output format
def outputBackchannelGauss(reader: DBReader, utt: str, uttInfo: DBEntry):
    back_channel_convid = uttInfo['convid']
    speaking_channel_convid = swap_speaker(back_channel_convid)
    radius_sec = 1
    peak = reader.getBCMaxTime(utt)

    input = reader.features.get_combined_feature(speaking_channel_convid)
    (frameN, coeffN) = input.shape

    output = reader.get_gauss_bc_feature(back_channel_convid)

    left_bound = reader.features.time_to_sample_index(input, peak - radius_sec)
    right_bound = reader.features.time_to_sample_index(input, peak + radius_sec)

    yield from np.append(input[left_bound:right_bound], output[left_bound:right_bound], axis=1)


def bc_to_category(reader: DBReader, uttInfo):
    if reader.extract_config.get("categories", None) is not None:
        txt = reader.noise_filter(uttInfo['text'].lower())
        outnum = reader.category_to_index[reader.bc_to_category[txt]]
        if outnum == 0:
            raise Exception("new phone who dis")
        return outnum
    else:
        return 1


def outputBackchannelDiscrete(reader: DBReader, utt: str, bc: bool) -> Tuple[np.array, np.array]:
    uttInfo = reader.uttDB[utt]
    back_channel_convid = uttInfo['convid']
    speaking_channel_convid = swap_speaker(back_channel_convid)
    begin, end = reader.getBackchannelTrainingRange(utt) if bc else reader.getNonBackchannelTrainingRange(utt)

    if begin < 0:
        logging.debug(
            "DEBUG: Skipping utt {}({})-, not enough data ({}s - {}s)".format(utt, uttInfo['text'], begin, end))
        return

    F = reader.features.get_combined_feature(speaking_channel_convid, begin, end + 0.1)
    range_frames = round((end - begin) * 1000 / F.frame_shift_ms)
    F = F[0:range_frames]
    frames, dim = F.shape
    if frames < range_frames:
        logging.debug(
            f"skipping utterance with F too small: {utt}({uttInfo['text']}): {frames}<{range_frames} ({begin}s - {end}s) (probably at end of file?)")
        return

    if bc:
        outnum = bc_to_category(reader, uttInfo)
    else:
        outnum = 0
    out = np.array([[outnum]], dtype=np.int32)
    return np.copy(F), np.repeat(out, frames, axis=0)


def read_conversations(config):
    return {name: list(parse_conversations_file(path)) for name, path in
            config['paths']['conversations'].items()}


def parseConversations(speaker: str, reader: DBReader):
    utts = list(reader.get_utterances(speaker))
    for index, (utt, uttInfo) in enumerate(utts):
        if not reader.is_backchannel(uttInfo, index, utts):
            continue

        yield from outputBackchannelDiscrete(reader, utt, uttInfo)


def parseConversationsList(speaker: str, config_path: str):
    reader = loadDBReader(config_path)
    return list(parseConversations(speaker, reader))


def parseConversationSet(parallel, config_path: str, setname: str, convIDs: Set[str]):
    logging.debug("parseConversationSet(" + setname + ")")
    reader = loadDBReader(config_path)
    speakers = list(speaker for speaker in reader.spkDB if speakerFilter(convIDs, speaker))
    for x in parallel(tqdm([delayed(parseConversationsList)(speaker, config_path) for speaker in speakers])):
        yield from x


@functools.lru_cache(maxsize=8)
def loadDBReader(config_path: str):
    return DBReader(config_path)


class UttDB:
    alignment_db = None

    def __init__(self, config: dict):
        paths_config = config['paths']
        allconvs = [conv for ls in read_conversations(config).values() for conv in ls]
        self.speakers = sorted([f"{conv}-{channel}" for conv in allconvs for channel in ["A", "B"]])
        self.root = paths_config['originalSwbTranscriptions']
        self.extractname = re.compile(r'sw(\d{4})([AB])-ms98-a-(\d{4})')

    @staticmethod
    @functools.lru_cache(maxsize=None)
    @util.DiskCache
    def load_utterances(root: str, convid: str, single_words=False):
        utts = OrderedDict()
        track = convid[2:6]
        speaker = convid[-1]
        type = "word" if single_words else "trans"
        id_counter = 0
        last_id = None
        with open(os.path.join(root, track[:2], track,
                               "sw{}{}-ms98-a-{}.text".format(track, speaker, type))) as file:
            for line in file:
                id, _from, to, text = line.split(maxsplit=3)
                if single_words:
                    if id == last_id:
                        id_counter += 1
                    else:
                        id_counter = 0
                    last_id = id
                    id += '-{}'.format(id_counter)
                utts[id] = {
                    'from': _from, 'to': to, 'text': text.strip(), 'convid': convid
                }
        return utts

    def get_speakers(self):
        return self.speakers

    def get_utterances(self, id: str) -> Iterable[Tuple[str, DBEntry]]:
        return self.load_utterances(self.root, id).items()

    def get_words_for_utterance(self, uttid: str, utt: DBEntry) -> List[DBEntry]:
        fromTime = float(utt['from'])
        toTime = float(utt['to'])
        words = self.load_utterances(self.root, utt['convid'], True).items()
        list = [word for id, word in words if float(word['from']) >= fromTime and float(word['to']) <= toTime]
        return list

    def __getitem__(self, id):
        track, speaker, uttid = self.extractname.fullmatch(id).groups()
        utts = self.load_utterances(self.root, f"sw{track}-{speaker}")
        return utts[id]


def parse_conversations_file(path: str):
    with open(path) as f:
        return set([line.strip() for line in f.readlines()])


def group(ids):
    for k, v in itertools.groupby(enumerate(ids), lambda x: x[1]):
        v2 = list(v)
        start_index, _ = v2[0]
        end_index, _ = v2[-1]
        yield start_index, k, end_index + 1


def bc_is_while_monologuing(reader: DBReader, uttInfo):
    fromTime = float(uttInfo['from'])
    toTime = float(uttInfo['to'])
    min_talk_len = reader.extract_config.get('min_talk_len', None)
    if min_talk_len is not None:
        middle = (toTime - fromTime) / 2
        bc_convid = uttInfo['convid']
        talking_convid = bc_convid[:-1] + dict(A="B", B="A")[bc_convid[-1]]
        filtered = list(util.filter_ranges([middle],
                                           list(
                                               util.get_monologuing_segments(reader, talking_convid, min_talk_len))))
        if filtered != [middle]:
            return False
        else:
            return True
    else:
        return True


def all_uttids_(config_path: str, convos: List[str]):
    # logging.debug(f"getting all bc uttids...{config_path}")
    reader = loadDBReader(config_path)
    for convo in convos:
        for channel in ["A", "B"]:
            convid = f"{convo}-{channel}"
            for (uttId, uttInfo) in reader.get_backchannels(list(reader.get_utterances(convid))):
                if bc_is_while_monologuing(reader, uttInfo):
                    yield uttId, False
                    yield uttId, True


# @functools.lru_cache(maxsize=8)
def all_uttids(config_path: str, convos: List[str]):
    return list(all_uttids_(config_path, convos))


def extract_convo(config_path: str, convo: str):
    reader = loadDBReader(config_path)
    out_dict = {}
    for uttId, is_bc in all_uttids(config_path, [convo]):
        out_dict[uttId, is_bc] = outputBackchannelDiscrete(reader, uttId, is_bc)
    return out_dict


def sample_m_of_n(m: int, things: list):
    n = len(things)
    while m > n:
        yield from things
        m -= n
    yield from random.sample(things, m)


# duplicate samples of categories with few samples until all categories have same number of samples if outputting
# categorical, else dont do anything
def balance_data(config_path: str, bcs: Iterable[Tuple[str, bool]]) -> Iterable[Tuple[str, int, bool]]:
    reader = loadDBReader(config_path)
    if reader.extract_config.get('categories', None) is not None:
        logging.info("balancing data by duplicating...")
        cat_to_uttids = {}
        for utt, is_bc in bcs:
            cat = bc_to_category(reader, reader.uttDB[utt]) if is_bc else 0
            cat_to_uttids.setdefault(cat, []).append((utt, is_bc))
        counts = {category: len(bcs) for category, bcs in cat_to_uttids.items()}
        pprint({reader.index_to_category.get(category, "NBC"): len(bcs) for category, bcs in cat_to_uttids.items()})
        max_count = max(counts.values())
        for category, utts in cat_to_uttids.items():
            yield from ((utt_id, index, is_bc) for index, (utt_id, is_bc) in enumerate(sample_m_of_n(max_count, utts)))
    else:
        yield from ((utt_id, index, is_bc) for index, (utt_id, is_bc) in enumerate(bcs))


# todo: code dedup with balance_data
# get a mapper that assigns every utt_id a weight for use in lasagne.objectives.aggregate(weights=_)
def get_balanced_weights(config_path: str, bcs: Iterable[Tuple[str, bool]]) -> Iterable[Tuple[str, float, bool]]:
    reader = loadDBReader(config_path)
    if reader.extract_config.get('categories', None) is not None:
        logging.info("balancing data by setting weights...")
        cat_to_uttids = {}
        for utt, is_bc in bcs:
            cat = bc_to_category(reader, reader.uttDB[utt]) if is_bc else 0
            cat_to_uttids.setdefault(cat, []).append((utt, is_bc))
        counts = {category: len(bcs) for category, bcs in cat_to_uttids.items()}
        pprint({reader.index_to_category.get(category, "NBC"): len(bcs) for category, bcs in cat_to_uttids.items()})
        max_count = max(counts.values())
        for category, utts in cat_to_uttids.items():
            yield from ((utt_id, max_count / counts[category], is_bc) for (utt_id, is_bc) in utts)
    else:
        yield from ((utt_id, 1.0, is_bc) for (utt_id, is_bc) in bcs)


config_path = None


@functools.lru_cache(maxsize=1)
@util.DiskCache
def pure_extract(extract_config: dict):
    config = util.load_config(config_path)
    logging.debug(f"extracting and saving data")
    convo_map = read_conversations(config)
    allconvos = [convo for convos in convo_map.values() for convo in convos]
    out_dict = {}
    for out in Parallel(n_jobs=int(os.environ.get('JOBS', -1)))(
            tqdm([delayed(extract_convo)(config_path, convo) for convo in allconvos])):
        out_dict.update(out)
    return out_dict


def extract(_config_path: str) -> Dict[Tuple[str, bool], Tuple[np.array, np.array]]:
    global config_path
    config_path = _config_path
    return pure_extract(util.hashabledict(util.load_config(config_path)['extract_config']))


@functools.lru_cache(maxsize=10)
def get_all_nonsilent_words(reader, convid):
    utts = list(reader.get_utterances(convid))
    return [word for utt, uttInfo in utts for word in reader.uttDB.get_words_for_utterance(utt, uttInfo) if
            len(reader.noise_filter(word['text'])) > 0]


def get_utterance_before(reader: DBReader, convid: str, time: float):
    import bisect
    words = get_all_nonsilent_words(reader, convid)
    keys = [float(word['to']) for word in words]
    inx = bisect.bisect_left(keys, time)
    if inx == 0:
        print(f"warning: no utt before {time} in {convid}")
        return words[0]
    return words[inx - 1]


def count(config_path):
    print(f"running frequency analysis")
    reader = loadDBReader(config_path)
    allconvs = [conv for ls in read_conversations(reader.config).values() for conv in ls]
    uttcount = 0
    uttwords = 0
    bccount = 0
    bcwords = 0
    bcdelays = []
    alltexts = []

    def word_count(utts):
        return len([word for (utt, uttInfo) in utts for word in orig_noise_filter(uttInfo['text']).split(" ")])

    for conv in allconvs:
        for channel in ["A", "B"]:
            convid = f"{conv}-{channel}"
            speaker_convid = f"{conv}-{util.invert_channel(channel)}"
            utts = list(reader.get_utterances(convid))
            uttwords += word_count(utts)
            bcs = list(reader.get_backchannels(utts))
            for bc, bcInfo in bcs:
                fromTime = reader.getBcRealFirstEndTime(bc)
                uttInfo = get_utterance_before(reader, speaker_convid, fromTime)
                lastTime = float(uttInfo['to'])
                bcdelays.append((bc, fromTime - lastTime))
                alltexts.append(reader.noise_filter(bcInfo['text']))
            bcwords += word_count(bcs)
            uttcount += len(utts)
            bccount += len(bcs)
    print(f"{bccount} backchannels out of a total of {uttcount} utterances ({bccount/uttcount*100:.3g}%) or " +
          f"{bcwords} words out of {uttwords} ({bcwords/uttwords*100:.3g}%)")
    print(f"most common: {Counter(alltexts).most_common(10)}")
    # print(f"\ndelays are")
    bcdelays.sort(key=lambda x: x[1])
    # with open("x", "w") as f:
    #    f.writelines([str(delay)+"\n" for utt, delay in bcdelays])

    import scipy.stats
    print(f"delay stats: {scipy.stats.describe([delay for uttid, delay in bcdelays])}")


@functools.lru_cache(maxsize=1)
def word_to_vec(config_path: str, dimension: int):
    folder = "data/word2vec"
    words_file = os.path.join(folder, f"words-noisefiltered-{dimension}")
    phrases_file = os.path.join(folder, f"phrases-noisefiltered-{dimension}")
    w2v_file = os.path.join(folder, f"noisefiltered-{dimension}.bin")
    import word2vec
    if os.path.isfile(w2v_file):
        logging.debug(f"{w2v_file} exists, returning that")
        return word2vec.load(w2v_file)

    reader = loadDBReader(config_path)
    allconvs = [conv for ls in read_conversations(reader.config).values() for conv in ls]

    def gen():
        for conv in tqdm(allconvs):
            for channel in ["A", "B"]:
                convid = f"{conv}-{channel}"
                for utt, uttInfo in reader.get_utterances(convid):
                    for word in reader.uttDB.get_words_for_utterance(utt, uttInfo):
                        txt = reader.noise_filter(word['text'])
                        if len(txt) > 0:
                            yield word['text'] + " "

    with open(words_file, "w") as f:
        f.writelines(gen())
    word2vec.word2phrase(words_file, phrases_file, verbose=True)
    word2vec.word2vec(phrases_file, w2v_file, size=dimension)
    logging.info("wrote to " + w2v_file)
    return word2vec.load(w2v_file)


def main():
    config_path = sys.argv[1]
    logging.root.handlers.clear()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[
                            logging.StreamHandler()
                        ])
    extract(config_path)


if __name__ == "__main__":
    main()
