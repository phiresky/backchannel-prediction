#!/usr/bin/env python3

# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/readDB.tcl
# (sha1sum 0e05e903997432917d5ef5b2ef4d799e196f3ffc)
# on 2016-11-06

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


def orig_noise_filter(text):
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

    # BC prediction area relative to BC start
    BCcenter = -0.3
    BCbegin = BCcenter - 0.2
    BCend = BCcenter + 0.2

    # Non BC prediction area relative to BC start
    NBCbegin = -2.3
    NBCend = -1.9

    noise_filter = None
    spkDB = None
    uttDB = None

    def __init__(self, config: Dict, config_path: str):
        self.config = config
        self.extract_config = self.config['extract_config']
        self.paths_config = self.config['paths']
        self.use_original_db = self.extract_config['useOriginalDB']
        self.features = Features(config, config_path)
        self.load_db()
        self.backchannels = load_backchannels(self.paths_config['backchannels'])
        self.sample_window_ms = self.extract_config['sample_window_ms']

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.spkDB.close()
        self.uttDB.close()

    def is_backchannel(self, uttInfo: dict, index: int, utts: List[Tuple[str, DBEntry]]):
        uttText = uttInfo['text']
        uttText = self.noise_filter(uttText)
        lastUttText = utts[index - 1][1]['text']
        lastUttText = self.noise_filter(lastUttText)
        return (uttText.lower() in self.backchannels and
                index > 0 and
                len(lastUttText) == 0
                )

    def get_gauss_bc_feature(self, convid: str):
        return self.get_gauss_bcs(self.features.get_power(convid), convid)

    def getBcRealStartTime(self, utt: str):
        if self.use_original_db:
            for word in self.uttDB.get_words_for_utterance(self.uttDB[utt]):
                text = self.noise_filter(word['text'])
                if len(text) > 0:
                    return float(word['from'])
        else:
            return self.getBCMaxTime(utt) - 0.3

    def get_max_time(self, feat: NumFeature, fromTime: float, toTime: float):
        powerrange = self.features.cut_range(feat, fromTime, toTime)
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
        return bc_start_time + self.BCbegin, bc_start_time + self.BCend

    def getNonBackchannelTrainingRange(self, utt: str):
        bc_start_time = self.getBcRealStartTime(utt)
        return bc_start_time + self.NBCbegin, bc_start_time + self.NBCend

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
        middle = self.getBCMaxTime(utt) + self.BCcenter
        width = 1.8
        times = np.arange(-width, width, power.shift / 1000, dtype="float32")
        mean = 0
        stddev = 0.3
        variance = stddev ** 2
        return middle - width, (1 / np.sqrt(2 * variance * np.pi)) * np.exp(-((times - mean) ** 2) / (2 * variance))

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

    def get_filtered_speakers(self, convIDs):
        return [spkr for spkr in self.spkDB if speakerFilter(convIDs, spkr)]

    def count_total(self, convIDs):
        l = list(bc for spkr in self.get_filtered_speakers(convIDs) for bc in
                 self.get_backchannels(list(self.get_utterances(spkr))))
        return len(l)


counter = 0
lastTime = time.clock()
input_dim = None


def load_backchannels(path):
    with open(path) as f:
        bcs = set([line.strip() for line in f.readlines()])
    return backchannels_hardcoded | bcs


def swap_speaker(convid: str):
    (conv, speaker) = convid.split("-")
    other_channel = dict(A="B", B="A")[speaker]
    return conv + "-" + other_channel


def outputBackchannelGauss(reader: DBReader, utt: str, uttInfo: DBEntry):
    back_channel_convid = uttInfo['convid']
    speaking_channel_convid = swap_speaker(back_channel_convid)
    radius_sec = 1
    peak = reader.getBCMaxTime(utt)

    input = reader.features.get_combined_feat(speaking_channel_convid)
    (frameN, coeffN) = input.shape

    output = reader.get_gauss_bc_feature(back_channel_convid)

    if coeffN != input_dim:
        raise Exception("coeff and dim don't match")

    left_bound = reader.features.time_to_sample_index(input, peak - radius_sec)
    right_bound = reader.features.time_to_sample_index(input, peak + radius_sec)

    yield from np.append(input[left_bound:right_bound], output[left_bound:right_bound], axis=1)


def outputBackchannelDiscrete(reader: DBReader, utt: str, uttInfo: DBEntry):
    back_channel_convid = uttInfo['convid']
    speaking_channel_convid = swap_speaker(back_channel_convid)
    cBCbegin, cBCend = reader.getBackchannelTrainingRange(utt)
    cNBCbegin, cNBCend = reader.getNonBackchannelTrainingRange(utt)

    fromTime = cNBCbegin - 1
    toTime = cBCend + 1
    if fromTime < 0:
        logging.debug(
            "DEBUG: Skipping utt {}({})-, not enough data ({}s - {}s)".format(utt, uttInfo['text'], fromTime,
                                                                              toTime))
        return

    F = reader.features.get_combined_feat(speaking_channel_convid)
    F = reader.features.cut_range(F, fromTime, toTime)
    (frameN, coeffN) = F.shape

    if coeffN != input_dim:
        raise Exception("coeff={} and dim={} don't match".format(coeffN, input_dim))

    expectedNumOfFrames = (toTime - fromTime) * 100
    deltaFrames = abs(expectedNumOfFrames - frameN)
    # logging.debug("deltaFrames %d", deltaFrames)
    if deltaFrames > 10:
        logging.warning("Frame deviation too big! expected {}, got {}".format(expectedNumOfFrames, frameN))
        return

    NBCframeStart = int((cNBCbegin - fromTime) * 100)
    NBCframeEnd = int((cNBCend - fromTime) * 100)
    BCframeStart = int((cBCbegin - fromTime) * 100)
    BCframeEnd = int((cBCend - fromTime) * 100)
    frameCount = 0
    for frameX in range(NBCframeStart, NBCframeEnd):
        yield np.append(F[frameX], [0], axis=0)
        frameCount += 1

    frameCount = 0
    for frameX in range(BCframeStart, BCframeEnd):
        yield np.append(F[frameX], [1], axis=0)
        frameCount += 1


def parseConversations(speaker: str, reader: DBReader):
    global counter, lastTime
    utts = list(reader.get_utterances(speaker))
    for index, (utt, uttInfo) in enumerate(utts):
        if not reader.is_backchannel(uttInfo, index, utts):
            continue

        yield from outputBackchannelDiscrete(reader, utt, uttInfo)

        counter += 1
        if counter % 100 == 0:
            took = time.clock() - lastTime
            lastTime = time.clock()
            logging.info("Written elements: %d (%.3fs per element)", counter, took / 100)


def parseConversationSet(reader: DBReader, setname: str, convIDs: Set[str]):
    logging.debug("parseConversationSet(" + setname + ")")
    speakers = list(speaker for speaker in reader.spkDB if speakerFilter(convIDs, speaker))
    for (i, speaker) in enumerate(speakers):
        logging.debug("parseConversations({}, {}) [{}/{}]".format(setname, speaker, i, len(speakers)))
        yield from parseConversations(speaker, reader)


def load_config(path):
    with open(path) as config_file:
        return json.load(config_file, object_pairs_hook=OrderedDict)


class FakeUttDB:
    alignment_db = None

    def __init__(self, paths_config, single_words=False):
        self.paths_config = paths_config
        self.root = paths_config['originalSwbTranscriptions']
        self.extractname = re.compile(r'sw(\d{4})([AB])-ms98-a-(\d{4})')
        self.spkDB = jrtk.base.DBase(baseFilename=paths_config['databasePrefix'] + "-spk", mode="r")
        self.single_words = single_words

    def makeSpkDB(self):
        class FakeSpkDB():
            uttDB = self

            def __iter__(self2):
                yield from self.spkDB

            def close(self):
                pass

        return FakeSpkDB()

    @functools.lru_cache(maxsize=10)
    def load_utterances(self, track: str, speaker: str):
        utts = OrderedDict()
        convid = 'sw{}-{}'.format(track, speaker)
        type = "word" if self.single_words else "trans"
        id_counter = 0
        last_id = None
        with open(os.path.join(self.root, track[:2], track,
                               "sw{}{}-ms98-a-{}.text".format(track, speaker, type))) as file:
            for line in file:
                id, _from, to, text = re.split("\s+", line, maxsplit=3)
                if self.single_words:
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
        yield from self.spkDB

    def get_utterances(self, id: str):
        return self.load_utterances(id[2:6], id[-1]).items()

    def get_words_for_utterance(self, utt: DBEntry) -> List[DBEntry]:
        if self.single_words:
            raise Exception("run this on the utts instance")
        if not self.alignment_db:
            self.alignment_db = FakeUttDB(self.paths_config, True)
        fromTime = float(utt['from'])
        toTime = float(utt['to'])
        words = self.alignment_db.get_utterances(utt['convid'])
        return [word for id, word in words if float(word['from']) >= fromTime and float(word['to']) <= toTime]

    def __getitem__(self, id):
        track, speaker, uttid = self.extractname.fullmatch(id).groups()
        utts = self.load_utterances(track, speaker)
        return utts[id]

    def close(self):
        self.spkDB.close()


def parse_conversations_file(path: str):
    with open(path) as f:
        return set([line.strip() for line in f.readlines()])


def main():
    global input_dim
    np.seterr(all='raise')
    logging.debug("loading config file {}".format(sys.argv[1]))
    config_path = sys.argv[1]
    config = load_config(config_path)

    extract_config = config['extract_config']
    context_ms = extract_config['context_ms']
    version = subprocess.check_output("git describe --dirty", shell=True).decode('ascii').strip()
    outputDir = os.path.join(extract_config['outputDirectory'], "{}".format(version))
    if os.path.isdir(outputDir):
        print("Output directory {} already exists, aborting".format(outputDir))
        sys.exit(1)
    logging.debug("outputting to " + outputDir)
    mkpath(outputDir)

    jrtk.core.setupLogging(os.path.join(outputDir, "extractBackchannels.log"), logging.DEBUG, logging.DEBUG)

    with DBReader(config, config_path) as reader:

        input_dim = 2 * context_ms // 10 / extract_config['context_stride']
        if not input_dim.is_integer():
            raise Exception("input dim is not integer: " + str(input_dim))
        output_dim = 1

        nnConfig = {
            'input_dim': int(input_dim),
            'output_dim': output_dim,
            'num_labels': 2,
            'files': {}
        }
        for setname, path in config['paths']['conversations'].items():
            convIDs = parse_conversations_file(path)
            # print("bc counts for {}: {}".format(setname, reader.count_total(convIDs)))
            data = fromiter(parseConversationSet(reader, setname, convIDs),
                            dtype="float32", shape=(-1, input_dim + output_dim))
            fname = os.path.join(outputDir, setname + ".npz")
            np.savez_compressed(fname, data=data)
            nnConfig['files'][setname] = os.path.relpath(os.path.abspath(fname), outputDir)

        jsonPath = os.path.join(outputDir, "config.json")
        with open(jsonPath, "w") as f:
            json.dump({**config, 'train_config': nnConfig}, f, indent='\t')
        logging.info("Wrote training config to " + os.path.abspath(jsonPath))


if __name__ == "__main__":
    main()
