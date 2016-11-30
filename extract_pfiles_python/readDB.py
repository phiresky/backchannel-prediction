#!/usr/bin/env python3

# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/readDB.tcl
# (sha1sum 0e05e903997432917d5ef5b2ef4d799e196f3ffc)
# on 2016-11-06

import jrtk
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


class DBReader:
    backchannels = None

    # BC area
    BCbegin = -0.75
    BCend = -0.35

    # Non BC area
    NBCbegin = -2.3
    NBCend = -1.9

    noise_filter = None
    spkDB = None
    uttDB = None

    def __init__(self, config):
        self.config = config
        self.extract_config = self.config['extract_config']
        self.paths_config = self.config['paths']
        self.context = self.extract_config['context']
        self.load_db()
        self.backchannels = load_backchannels(self.paths_config['backchannels'])
        self.feature_extractor = load_feature_extractor(config)
        self.sample_window_ms = self.extract_config['sample_window_ms']

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.spkDB.close()
        self.uttDB.close()

    def speakerFilter(self, convIDs: Set[str], speaker: str) -> bool:
        shortID = speaker.split("-")[0]
        return shortID in convIDs

    def is_backchannel(self, uttInfo: dict, index: int, utts: List[Tuple[str, DBEntry]]):
        uttText = uttInfo['text']
        uttText = self.noise_filter(uttText)
        lastUttText = utts[index - 1][1]['text']
        lastUttText = self.noise_filter(lastUttText)
        return (uttText.lower() in self.backchannels and
                index > 0 and
                len(lastUttText) == 0
                )

    def sample_index_to_time(self, feat: jrtk.preprocessing.NumFeature, sample_index):
        if feat.typ != jrtk.features.FeatureType.FMatrix:
            raise Exception("only for extracted features")
        return (self.sample_window_ms / 2 + sample_index * feat.shift) / 1000

    def time_to_sample_index(self, feat: jrtk.preprocessing.NumFeature, time_sec):
        if feat.typ != jrtk.features.FeatureType.FMatrix:
            raise Exception("only for extracted features")
        return int(round((1000 * time_sec - self.sample_window_ms / 2) / feat.shift))

    @functools.lru_cache(maxsize=2)
    def getBCMaxTime(self, utt: str):
        uttInfo = self.uttDB[utt]
        (audiofile, channel) = uttInfo['convid'].split("-")
        uttFrom = float(uttInfo['from'])
        uttTo = float(uttInfo['to'])
        fromTime = max(0, uttFrom - 1)
        toTime = uttTo + 1
        features = self.feature_extractor.eval(None, {
            'from': fromTime,
            'to': toTime,
            'conv': audiofile
        })
        power = features['power' + channel.lower()]
        powerrange = power[self.time_to_sample_index(power, uttFrom - fromTime):self.time_to_sample_index(power, uttTo - fromTime)]
        maxIndex = powerrange.argmax()
        maxTime = uttFrom + self.sample_index_to_time(powerrange, maxIndex)
        return maxTime

    def getBackchannelTrainingRange(self, utt: str):
        fromTime = self.getBCMaxTime(utt)
        return fromTime + self.BCbegin, fromTime + self.BCend

    def getNonBackchannelTrainingRange(self, utt: str):
        fromTime = self.getBCMaxTime(utt)
        return fromTime + self.NBCbegin, fromTime + self.NBCend

    def load_db(self) -> (DBase, DBase):
        if self.config['extract_config']['useOriginalDB']:
            uttDB = FakeUttDB(self.paths_config)
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

    def get_filtered_speakers(self, convIDs):
        return [spkr for spkr in self.spkDB if self.speakerFilter(convIDs, spkr)]

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


def parseConversations(speaker: str, reader: DBReader):
    global counter, lastTime
    utts = list(reader.get_utterances(speaker))
    for index, (utt, uttInfo) in enumerate(utts):
        convID = uttInfo['convid']
        (audiofile, channel) = convID.split("-")
        toTime = float(uttInfo['to'])
        fromTime = float(uttInfo['from'])
        if not reader.is_backchannel(uttInfo, index, utts):
            continue
        # print('has backchannel: ' + uttInfo['text'])
        cBCbegin, cBCend = reader.getBackchannelTrainingRange(utt)
        cNBCbegin, cNBCend = reader.getNonBackchannelTrainingRange(utt)

        fromTime = cNBCbegin - 1
        toTime = cBCend + 1
        if fromTime < 0:
            logging.debug(
                "DEBUG: Skipping utt {}({})-, not enough data ({}s - {}s)".format(utt, uttInfo['text'], fromTime,
                                                                                  toTime))
            continue

        if channel == "A":
            BCchannel = "B"
        elif channel == "B":
            BCchannel = "A"
        else:
            raise Exception("Unknown channel " + channel)


        features = reader.feature_extractor.eval(None, {
            'from': fromTime,
            'to': toTime,
            'conv': audiofile
        })
        F = features["feat" + BCchannel.lower()]
        (frameN, coeffN) = F.shape

        if coeffN != input_dim:
            raise Exception("coeff and dim don't match")

        expectedNumOfFrames = (toTime - fromTime) * 100
        deltaFrames = abs(expectedNumOfFrames - frameN)
        # logging.debug("deltaFrames %d", deltaFrames)
        if deltaFrames > 10:
            logging.warning("Frame deviation too big!")
            continue

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

        counter += 1
        if counter % 100 == 0:
            took = time.clock() - lastTime
            lastTime = time.clock()
            logging.info("Written elements: %d (%.3fs per element)", counter, took / 100)


def parseConversationSet(reader: DBReader, setname: str, convIDs: Set[str]):
    logging.debug("parseConversationSet(" + setname + ")")
    speakers = list(speaker for speaker in reader.spkDB if reader.speakerFilter(convIDs, speaker))
    for (i, speaker) in enumerate(speakers):
        logging.debug("parseConversations({}, {}) [{}/{}]".format(setname, speaker, i, len(speakers)))
        yield from parseConversations(speaker, reader)


def load_config(path):
    with open(path) as config_file:
        return json.load(config_file, object_pairs_hook=OrderedDict)


def load_feature_extractor(config):
    featureSet = jrtk.preprocessing.FeatureExtractor(config=config)
    for step in config['extract_config']['featureExtractionSteps']:
        featureSet.appendStep(step)
    return featureSet


class FakeUttDB:
    def __init__(self, paths_config):
        self.root = paths_config['originalSwbTranscriptions']
        self.extractname = re.compile(r'sw(\d{4})([AB])-ms98-a-(\d{4})')
        self.spkDB = jrtk.base.DBase(baseFilename=paths_config['databasePrefix'] + "-spk", mode="r")

    def makeSpkDB(self):
        class FakeSpkDB():
            uttDB = self

            def __iter__(self2):
                yield from self.spkDB

            def close(self):
                pass

        return FakeSpkDB()

    @functools.lru_cache(maxsize=5)
    def load_utterances(self, track: str, speaker: str):
        utts = OrderedDict()
        convid = 'sw{}-{}'.format(track, speaker)
        with open(os.path.join(self.root, track[:2], track, "sw{}{}-ms98-a-trans.text".format(track, speaker))) as file:
            for line in file:
                id, _from, to, text = line.split(" ", 3)
                utts[id] = {
                    'from': _from, 'to': to, 'text': text.strip(), 'convid': convid
                }
        return utts

    def get_speakers(self):
        yield from self.spkDB

    def get_utterances(self, id: str):
        return self.load_utterances(id[2:6], id[-1]).items()

    def __getitem__(self, id):
        track, speaker, uttid = self.extractname.fullmatch(id).groups()
        utts = self.load_utterances(track, speaker)
        return utts[id]

    def close(self):
        self.spkDB.close()


def main():
    global input_dim
    np.seterr(all='raise')
    logging.debug("loading config file {}".format(sys.argv[1]))
    config = load_config(sys.argv[1])

    extract_config = config['extract_config']
    context = extract_config['context']
    version = subprocess.check_output("git describe --dirty", shell=True).decode('ascii').strip()
    outputDir = os.path.join(extract_config['outputDirectory'],
                             "{}-context{}".format(version, context))
    if os.path.isdir(outputDir):
        print("Output directory {} already exists, aborting".format(outputDir))
        sys.exit(1)
    logging.debug("outputting to " + outputDir)
    mkpath(outputDir)

    jrtk.core.setupLogging(os.path.join(outputDir, "extractBackchannels.log"), logging.DEBUG, logging.DEBUG)

    with DBReader(config) as reader:

        input_dim = 2 * (context * 2 + 1)
        output_dim = 1

        nnConfig = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'num_labels': 2,
            'files': {}
        }
        for setname, path in config['paths']['conversations'].items():
            with open(path) as f:
                convIDs = set([line.strip() for line in f.readlines()])
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
