#!/usr/bin/env python3

# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/readDB.tcl
# (sha1sum 0e05e903997432917d5ef5b2ef4d799e196f3ffc)
# on 2016-11-06

import jrtk
from typing import Set, Dict, List
import logging
import os
from distutils.dir_util import mkpath
import numpy as np
import time
import sys
import json
import subprocess
from collections import OrderedDict


def speakerFilter(convIDs: Set[str], speaker: str) -> bool:
    shortID = speaker.split("-")[0]
    return shortID in convIDs


backchannels_hardcoded = {'hum', 'um-hum', 'hm'}  # these do not appear in the switchboard dialog act corpus
backchannels = None
DBase = Dict[str, Dict[str, str]]

# BC area
BCbegin = -0.4
BCend = 0.0

# Non BC area
NBCbegin = -2.0
NBCend = -1.6


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


def load_backchannels(config):
    global backchannels
    with open(config['paths']['backchannels']) as f:
        bcs = set([line.strip() for line in f.readlines()])
    backchannels = backchannels_hardcoded | bcs


def is_backchannel(uttInfo: dict, index: int, utts: List[str], uttDB: DBase):
    return (uttInfo['text'].lower() in backchannels
            and index > 0
            and uttDB[utts[index - 1]]['text'] == '<SIL>'
            )


def getBackchannelTrainingRange(uttInfo):
    fromTime = float(uttInfo['from'])
    return fromTime + BCbegin, fromTime + BCend


def getNonBackchannelTrainingRange(uttInfo):
    fromTime = float(uttInfo['from'])
    return fromTime + NBCbegin, fromTime + NBCend


counter = 0
lastTime = time.clock()
input_dim = None


def parseConversations(speaker: str, spkDB: DBase, uttDB: DBase, featureSet: jrtk.preprocessing.FeatureExtractor):
    global counter, lastTime
    utts = getUtterances(spkDB, speaker)
    for index, utt in enumerate(utts):
        uttInfo = uttDB[utt]
        convID = uttInfo['convid']
        (audiofile, channel) = convID.split("-")
        toTime = float(uttInfo['to'])
        fromTime = float(uttInfo['from'])
        if not is_backchannel(uttInfo, index, utts, uttDB):
            continue
        # print('has backchannel: ' + uttInfo['text'])
        cBCbegin, cBCend = getBackchannelTrainingRange(uttInfo)
        cNBCbegin, cNBCend = getNonBackchannelTrainingRange(uttInfo)
        if cNBCbegin < 0:
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

        fromTime = max(cNBCbegin - 1, 0)
        toTime = cBCend + 1
        features = featureSet.eval(None, {
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
        for frameX in range(NBCframeStart, NBCframeEnd + 1):
            yield np.append(F[frameX], [0], axis=0)
            frameCount += 1

        frameCount = 0
        for frameX in range(BCframeStart, BCframeEnd + 1):
            yield np.append(F[frameX], [1], axis=0)
            frameCount += 1

        counter += 1
        if counter % 100 == 0:
            took = time.clock() - lastTime
            lastTime = time.clock()
            logging.info("Written elements: %d (%.3fs per element)", counter, took / 100)


def parseConversationSet(spkDB: DBase, uttDB: DBase, setname: str, convIDs: Set[str], featureSet):
    logging.debug("parseConversationSet(" + setname + ")")
    speakers = list(speaker for speaker in spkDB if speakerFilter(convIDs, speaker))
    for (i, speaker) in enumerate(speakers):
        logging.debug("parseConversations({}, {}) [{}/{}]".format(setname, speaker, i, len(speakers)))
        yield from parseConversations(speaker, spkDB, uttDB, featureSet)


def load_config(path):
    with open(path) as config_file:
        return json.load(config_file, object_pairs_hook=OrderedDict)


def load_feature_extractor(config):
    featureSet = jrtk.preprocessing.FeatureExtractor(config=config)
    for step in config['extract_config']['featureExtractionSteps']:
        featureSet.appendStep(step)
    return featureSet


def load_db(paths_config):
    uttDB = jrtk.base.DBase(baseFilename=paths_config['databasePrefix'] + "-utt", mode="r")
    spkDB = jrtk.base.DBase(baseFilename=paths_config['databasePrefix'] + "-spk", mode="r")
    return spkDB, uttDB


def getUtterances(spkDB, spkr: str):
    return spkDB[spkr]['segs'].strip().split(" ")


def getBackchannels(uttDB, utts: List[str]):
    return [uttDB[utt]
            for index, utt in enumerate(utts)
            if is_backchannel(uttDB[utt], index, utts, uttDB)
            ]


def count_total(uttDB, spkDB, convIDs):
    l = list(bc for spkr in spkDB if speakerFilter(convIDs, spkr) for bc in
             getBackchannels(uttDB, getUtterances(spkDB, spkr)))
    return len(l)


def main():
    np.seterr(all='raise')
    global config, input_dim
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

    spkDB, uttDB = load_db(config['paths'])
    load_backchannels(config)

    featureSet = load_feature_extractor(config)

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
            print("bc counts for {}: {}".format(setname, count_total(uttDB, spkDB, convIDs)))
            # data = fromiter(parseConversationSet(spkDB, uttDB, setname, convIDs, featureSet),
            #                dtype="float32", shape=(-1, input_dim + output_dim))
            # fname = os.path.join(outputDir, setname + ".npz")
            # np.savez_compressed(fname, data=data)
            # nnConfig['files'][setname] = os.path.relpath(os.path.abspath(fname), outputDir)

    jsonPath = os.path.join(outputDir, "config.json")
    with open(jsonPath, "w") as f:
        json.dump({**config, 'train_config': nnConfig}, f, indent='\t')
    logging.info("Wrote training config to " + os.path.abspath(jsonPath))
    uttDB.close()
    spkDB.close()


if __name__ == "__main__":
    main()
