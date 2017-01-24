#!/usr/bin/env python3

# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/readDB.tcl
# (sha1sum 0e05e903997432917d5ef5b2ef4d799e196f3ffc)
# on 2016-11-06
import itertools
import jrtk
from jrtk.preprocessing import NumFeature
from typing import Set, Dict, List, Iterable, Tuple
import logging
import os
from distutils.dir_util import mkpath
import numpy as np
import time
import sys
import json
import subprocess
from collections import OrderedDict
import os.path
import re
import functools
from .features import Features
from tqdm import tqdm
from . import util
import hashlib

os.environ['JOBLIB_START_METHOD'] = 'forkserver'
from joblib import Parallel, delayed
import pickle

MAX_TIME = 100 * 60 * 60  # 100 hours
# these do not appear in the switchboard dialog act corpus but are very common in the sw98 transcriptions
# (counted with checkBackchannels.count_utterances)
backchannels_hardcoded = {'hum', 'um-hum', 'hm', 'yeah yeah', 'um-hum um-hum', 'uh-huh uh-huh',
                          'right right', 'right yeah', "yeah that's true", 'uh-huh yeah', 'um-hum yeah', 'okay_1',
                          'yeah right', 'yeah uh-huh', 'yeah well', 'yes yes', 'absolutely', 'right uh-huh',
                          }
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
    spkDB = None
    uttDB = None
    method = None

    def __init__(self, config: Dict, config_path: str):
        self.config = config
        self.extract_config = self.config['extract_config']
        self.paths_config = self.config['paths']
        self.use_original_db = self.extract_config['useOriginalDB']
        self.features = Features(config, config_path)
        self.load_db()
        self.backchannels = load_backchannels(self.paths_config['backchannels'])
        self.sample_window_ms = self.extract_config['sample_window_ms']
        self.method = self.extract_config['extraction_method']

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.spkDB.close()
        self.uttDB.close()

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

    def get_max_time(self, feat: NumFeature, fromTime: float, toTime: float):
        powerrange = self.features.cut_range_old(feat, fromTime, toTime)
        maxIndex = powerrange.argmax()
        maxTime = fromTime + self.features.sample_index_to_time(powerrange, maxIndex)
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
            uttDB = FakeUttDB(self.paths_config, self.extract_config['useWordsTranscript'])
            self.noise_filter = orig_noise_filter
            self.spkDB = uttDB.makeSpkDB()
            self.uttDB = uttDB
        else:
            self.noise_filter = isl_noise_filter
            self.uttDB = jrtk.base.DBase(baseFilename=self.paths_config['databasePrefix'] + "-utt", mode="r")
            self.spkDB = jrtk.base.DBase(baseFilename=self.paths_config['databasePrefix'] + "-spk", mode="r")

    def get_utterances(self, spkr: str) -> Iterable[Tuple[str, DBEntry]]:
        if not isinstance(self.spkDB, jrtk.base.DBase):
            return self.uttDB.get_utterances(spkr)
        else:
            return [(utt, self.uttDB[utt]) for utt in self.spkDB[spkr]['segs'].strip().split(" ")]

    def get_backchannels(self, utts: List[Tuple[str, DBEntry]]) -> List[Tuple[str, DBEntry]]:
        return [(utt, uttInfo)
                for index, (utt, uttInfo) in enumerate(utts)
                if self.is_backchannel(uttInfo, index, utts)
                ]

    def get_gauss_bc_array(self, power: NumFeature, utt: str) -> Tuple[float, np.array]:
        middle = self.getBcRealStartTime(utt) + self.method['peak']
        stddev = self.method['stddev']
        radius = self.method['radius']
        times = np.arange(-radius, radius, power.shift / 1000, dtype="float32")
        mean = 0
        variance = stddev ** 2
        return middle - radius, (1 / np.sqrt(2 * variance * np.pi)) * np.exp(-((times - mean) ** 2) / (2 * variance))

    def get_gauss_bcs(self, power_feature: np.array, spkr: str):
        arr = np.zeros_like(power_feature)
        feat = NumFeature(arr, shift=power_feature.shift)
        for bc, _ in self.get_backchannels(list(self.get_utterances(spkr))):
            offset, gauss = self.get_gauss_bc_array(power_feature, bc)
            gauss = gauss.reshape((gauss.shape[0], 1))
            start = self.features.time_to_sample_index(feat, offset)
            if start < 0:
                gauss = gauss[0 - start:]
                start = 0
            if len(gauss) > arr.size - start:
                gauss = gauss[:arr.size - start]
            feat[start:start + len(gauss)] += gauss
        return feat


def load_backchannels(path):
    with open(path) as f:
        bcs = set([line.strip() for line in f.readlines() if line[0] != '#'])
    return backchannels_hardcoded | bcs


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
    range_frames = int((end - begin) * 1000 / F.shift)
    F = F[0:range_frames]
    frames, dim = F.shape
    if frames < range_frames:
        raise Exception(f"FBC too small: {bc_frames}")

    out = np.array([[1 if bc else 0]], dtype=np.int32)
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


@functools.lru_cache(maxsize=1)
def loadDBReader(config_path: str):
    config = util.load_config(config_path)
    return DBReader(config, config_path)


class FakeUttDB:
    alignment_db = None
    uttsCache = {}
    wordsCache = {}

    def __init__(self, paths_config, single_words=False):
        self.paths_config = paths_config
        self.root = paths_config['originalSwbTranscriptions']
        self.extractname = re.compile(r'sw(\d{4})([AB])-ms98-a-(\d{4})')
        self.spkDB = jrtk.base.DBase(baseFilename=paths_config['databasePrefix'] + "-spk", mode="r")
        if os.path.isfile("data/uttdbcache.json"):
            with open("data/uttdbcache.json", "r") as f:
                x = json.load(f)
                self.uttsCache = x['uttsCache']
                self.wordsCache = x['wordsCache']
        else:
            logging.warning("uttdbcache does not exist, run create_uttdb_cache for speedup")

    def makeSpkDB(self):
        class FakeSpkDB():
            uttDB = self

            def __iter__(self2):
                yield from self.spkDB

            def close(self):
                pass

        return FakeSpkDB()

    @functools.lru_cache()
    def load_utterances(self, convid: str, single_words=False):
        utts = OrderedDict()
        track = convid[2:6]
        speaker = convid[-1]
        if not single_words and convid in self.uttsCache:
            return self.uttsCache[convid]
        type = "word" if single_words else "trans"
        id_counter = 0
        last_id = None
        with open(os.path.join(self.root, track[:2], track,
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
        if not single_words:
            self.uttsCache[convid] = utts
        return utts

    def get_speakers(self):
        yield from self.spkDB

    def get_utterances(self, id: str):
        return self.load_utterances(id).items()

    def get_words_for_utterance(self, uttid: str, utt: DBEntry) -> List[DBEntry]:
        if uttid in self.wordsCache:
            return self.wordsCache[uttid]
        fromTime = float(utt['from'])
        toTime = float(utt['to'])
        words = self.load_utterances(utt['convid'], True).items()
        list = [word for id, word in words if float(word['from']) >= fromTime and float(word['to']) <= toTime]
        self.wordsCache[uttid] = list
        return list

    def __getitem__(self, id):
        track, speaker, uttid = self.extractname.fullmatch(id).groups()
        utts = self.load_utterances(f"sw{track}-{speaker}")
        return utts[id]

    def close(self):
        self.spkDB.close()


def create_uttdb_cache(config_path: str):
    config = util.load_config(config_path)
    db = FakeUttDB(config['paths'])
    for spk in db.get_speakers():
        if not spk[0:2] == "sw":
            continue
        for utt, uttInfo in db.get_utterances(spk):
            db.get_words_for_utterance(utt, uttInfo)
    with open("data/uttdbcache.json", "w") as f:
        json.dump({"uttsCache": db.uttsCache, "wordsCache": db.wordsCache}, f)


def parse_conversations_file(path: str):
    with open(path) as f:
        return set([line.strip() for line in f.readlines()])


def group(ids):
    for k, v in itertools.groupby(enumerate(ids), lambda x: x[1]):
        v2 = list(v)
        start_index, _ = v2[0]
        end_index, _ = v2[-1]
        yield start_index, k, end_index + 1


def all_uttids(reader: DBReader, convos: List[str]):
    for convo in convos:
        for channel in ["A", "B"]:
            convid = f"{convo}-{channel}"
            for (uttId, uttInfo) in reader.get_backchannels(list(reader.get_utterances(convid))):
                yield uttId, False
                yield uttId, True


@functools.lru_cache(maxsize=1)
def extract(config_path: str) -> Dict[Tuple[str, bool], Tuple[np.array, np.array]]:
    config = util.load_config(config_path)
    extract_config = config['extract_config']
    meta = dict(extract_config=extract_config)
    meta_json = json.dumps(meta, sort_keys=True, indent='\t').encode('ascii')
    digest = hashlib.sha256(meta_json).hexdigest()
    path = os.path.join('data/cache', f"extract-{digest}.pickle")
    if os.path.exists(path):
        logging.debug(f"loading cached extracted data from {path}")
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        logging.debug(f"extracting and saving data to {path}")
        reader = DBReader(config, config_path)
        convos = read_conversations(config)
        c = [convo for convos in convos.values() for convo in convos]
        out_dict = {}
        for uttId, is_bc in tqdm(list(all_uttids(reader, c))):
            out_dict[uttId, is_bc] = outputBackchannelDiscrete(reader, uttId, is_bc)
        val = out_dict

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", 'wb') as file:
            pickle.dump(val, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(path + ".part", path)
        with open(path + '.meta.json', 'wb') as file:
            file.write(meta_json)
        return val


def main():
    config_path = sys.argv[1]
    jrtk.core.setupLogging(None, logging.DEBUG, logging.DEBUG)
    logging.root.handlers.clear()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[
                            logging.StreamHandler()
                        ])
    extract(config_path)


if __name__ == "__main__":
    main()
