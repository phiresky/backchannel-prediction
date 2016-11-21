#!/usr/bin/env python3

# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/readDB.tcl
# (sha1sum 0e05e903997432917d5ef5b2ef4d799e196f3ffc)
# on 2016-11-06

import jrtk
from typing import Set, Dict
import logging
import os
from distutils.dir_util import mkpath
import numpy as np
import time
import sys
import json
from collections import OrderedDict

def speakerFilter(convIDs: Set[str], speaker: str) -> bool:
    shortID = speaker.split("-")[0]
    return shortID in convIDs


backChannelListOneWord = {
    "UM-HUM", "UH-HUH", "YEAH", "RIGHT", "OH", "UM", "YES", "HUH", "OKAY", "HM", "HUM", "UH"
}

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


def isBackchannel(uttInfo):
    return uttInfo['text'] in backChannelListOneWord


def getBackchannelTrainingRange(uttInfo):
    fromTime = float(uttInfo['from'])
    return fromTime + BCbegin, fromTime + BCend


def getNonBackchannelTrainingRange(uttInfo):
    fromTime = float(uttInfo['from'])
    return fromTime + NBCbegin, fromTime + NBCend


config = None
counter = 0
lastTime = time.clock()
dim = None


def parseConversations(speaker: str, spkDB: jrtk.base.DBase, uttDB: jrtk.base.DBase,
                       featureSet: jrtk.preprocessing.FeatureExtractor):
    global counter, lastTime
    utts = spkDB[speaker]['segs'].strip().split(" ")
    for utt in utts:
        uttInfo = uttDB[utt]  # type: dict
        convID = uttInfo['convid']  # type: str
        (audiofile, channel) = convID.split("-")  # type: str
        toTime = float(uttInfo['to'])
        fromTime = float(uttInfo['from'])
        if not isBackchannel(uttInfo):
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

        fromTime = cNBCbegin
        toTime = cBCend
        features = featureSet.eval(None, {
            'from': fromTime,
            'to': toTime,
            'conv': audiofile
        })
        F = features["feat" + BCchannel.lower()]
        (frameN, coeffN) = F.shape

        if coeffN != dim:
            raise Exception("coeff and dim don't match")

        expectedNumOfFrames = (toTime - fromTime) * 100
        deltaFrames = abs(expectedNumOfFrames - frameN)
        # logging.debug("deltaFrames %d", deltaFrames)
        if deltaFrames > 10:
            logging.warning("Frame deviation too big!")
            continue

        NBCframeCount = int((cNBCend - cNBCbegin) * 100)
        BCframeCount = int((cBCend - cBCbegin) * 100)
        frameCount = 0
        for frameX in range(0, NBCframeCount):
            yield np.append(F[frameX], [0], axis=0)
            frameCount += 1

        frameCount = 0
        for frameX in range(frameN - BCframeCount, frameN):
            yield np.append(F[frameX], [1], axis=0)
            frameCount += 1

        counter += 1
        if counter % 100 == 0:
            took = time.clock() - lastTime
            lastTime = time.clock()
            logging.info("Written elements: %d (%.3fs per element)", counter, took / 100)


def parseConversationSet(spkDB: jrtk.base.DBase, uttDB: jrtk.base.DBase, setname: str, convIDs: Set[str], featureSet):
    logging.debug("parseConversationSet(" + setname + ")")
    speakers = list(speaker for speaker in spkDB if speakerFilter(convIDs, speaker))
    for (i, speaker) in enumerate(speakers):
        logging.debug("parseConversations({}, {}) [{}/{}]".format(setname, speaker, i, len(speakers)))
        yield from parseConversations(speaker, spkDB, uttDB, featureSet)


def main():
    np.seterr(all='raise')
    global config, dim
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file, object_pairs_hook=OrderedDict)

    context = config['context']
    outputDir = os.path.join(config['outputDirectory'], "context" + str(context))
    if os.path.isdir(outputDir):
        print("Output directory {} already exists, aborting".format(outputDir))
        sys.exit(1)
    mkpath(outputDir)

    jrtk.core.setupLogging(os.path.join(outputDir, "extractBackchannels.log"), logging.DEBUG, logging.DEBUG)

    uttDB = jrtk.base.DBase(baseFilename=config['databasePrefix'] + "-utt", mode="r")
    spkDB = jrtk.base.DBase(baseFilename=config['databasePrefix'] + "-spk", mode="r")

    featureSet = jrtk.preprocessing.FeatureExtractor(config=config)
    featureSet.appendStep('featAccess.py')
    featureSet.appendStep('featDescDelta.py')

    dim = 2 * (config['context'] * 2 + 1)

    nnConfig = {
        'input_dim': dim,
        'output_dim': 1,
        'num_labels': 2,
        'files': {}
    }
    for setname, path in config['conversations'].items():
        with open(path) as f:
            convIDs = set([line.strip() for line in f.readlines()])
        data = fromiter(parseConversationSet(spkDB, uttDB, setname, convIDs, featureSet),
                        dtype="float32", shape=(-1, dim + 2))
        fname = os.path.join(outputDir, setname + ".npz")
        np.savez_compressed(fname, data=data)
        nnConfig['files'][setname] = os.path.relpath(os.path.abspath(fname), outputDir)

    jsonPath = os.path.join(outputDir, "train-config.json")
    with open(jsonPath, "w") as f:
        json.dump(nnConfig, f, indent='\t')
    logging.info("Wrote training config to " + os.path.abspath(jsonPath))
    uttDB.close()
    spkDB.close()


if __name__ == "__main__":
    main()
