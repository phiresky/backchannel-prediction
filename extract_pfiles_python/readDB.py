#!/usr/bin/env python3
import jrtk
import argparse
from typing import Set
import logging
import os
from distutils.dir_util import mkpath
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)
logging.warning("test")


def dbaseUttFilter(convIDs: Set[str], convid: str) -> bool:
    # uttInfo = dbase[utt] # type: dict
    # convid = uttInfo['convid'] # type: str
    if "en_" in convid:
        logging.debug("Skipping utterance -> " + convid)
        return False

    shortID = convid.split("-")[0]
    if shortID not in convIDs:
        logging.debug("CONVID not on list -> " + convid)
        return False
    return True


backChannelListOneWord = {
    "UM-HUM", "UH-HUH", "YEAH", "RIGHT", "OH", "UM", "YES", "HUH", "OKAY", "HM", "HUM", "UH"
}

# BC area
BCbegin = -0.4
BCend = 0.0

# Non BC area
NBCbegin = -2.0
NBCend = -1.6

# Properties for writing pfiles
feature = "FEAT"  # Feature to be used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputDBase', type=str, help="name of the input database")
    parser.add_argument('pfilePrefix', type=str, help="prefix of written pfiles")
    parser.add_argument('convIdFile', type=str, help="file with allowed conv ids")
    parser.add_argument('-dataPath', type=str, help="Directory for pfiles")
    parser.add_argument('-delta', type=int, help="delta for input features")
    # parser.add_argument('--test', type=float)

    jrtk.core.setupLogging(None, None, logging.DEBUG)
    args = parser.parse_args()
    uttDB = jrtk.base.DBase(baseFilename=args.inputDBase + "-utt", mode="r")
    spkDB = jrtk.base.DBase(baseFilename=args.inputDBase + "-spk", mode="r")
    with open(args.convIdFile) as f:
        convIDs = set([line.strip() for line in f.readlines()])

    dim = 2 * (args.delta * 2 + 1)

    BCcounter = 0
    NBCcounter = 0
    counter = 0
    mkpath(args.dataPath)
    fpBC = open(os.path.join(args.dataPath, args.pfilePrefix + "-BC.txt"), "w")
    fpNBC = open(os.path.join(args.dataPath, args.pfilePrefix + "-NBC.txt"), "w")

    featureSetBC = jrtk.preprocessing.FeatureExtractor(config={'delta': args.delta, 'base': '../ears2/earsData'})
    featureSetBC.appendStep('featAccess.py')
    featureSetBC.appendStep('featDescDelta.py')
    lastTime = time.clock()
    for utt in uttDB:
        uttInfo = uttDB[utt]  # type: dict
        convID = uttInfo['convid']  # type: str
        if not dbaseUttFilter(convIDs, convID):
            continue
        channel = convID[-1]  # type: str
        audiofile = convID.split("-")[0]  # type: str
        uttInfo['channel'] = channel
        uttInfo['conv'] = audiofile
        toTime = float(uttInfo['to'])
        fromTime = float(uttInfo['from'])
        if uttInfo['text'] in backChannelListOneWord:
            # print('has backchannel: ' + uttInfo['text'])
            length = toTime - fromTime
            cBCbegin = fromTime + BCbegin
            cBCend = fromTime + BCend
            cNBCbegin = fromTime + NBCbegin
            cNBCend = fromTime + NBCend
            if cBCbegin < 1.0 or cNBCbegin < 0:
                logging.debug(
                    "DEBUG: Skipping utt {}({})-, not enough data ({}s - {}s".format(utt, uttInfo['text'], fromTime,
                                                                                     toTime))
                continue

            if channel == "A":
                BCchannel = "B"
            elif channel == "B":
                BCchannel = "A"
            else:
                raise Exception("Unknown channel " + channel)

            cFeature = (feature + BCchannel).lower()

            fromTime = cNBCbegin
            toTime = cBCend
            uttInfo['from'] = fromTime
            uttInfo['to'] = toTime
            features = featureSetBC.eval(None, uttInfo)
            F = features[cFeature]
            (frameN, coeffN) = features[cFeature].shape

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
                fpNBC.write("{} {}   {}  0\n".format(NBCcounter, frameCount, "  ".join(np.char.mod("%.6e", F[frameX]))))
                frameCount += 1
            NBCcounter += 1

            frameCount = 0
            for frameX in range(frameN - BCframeCount, frameN):
                fpBC.write("{} {}   {}  1\n".format(BCcounter, frameCount, "  ".join(np.char.mod("%.6e", F[frameX]))))
                frameCount += 1
            BCcounter += 1
            counter += 1
            if counter % 100 == 0:
                took = time.clock() - lastTime
                lastTime = time.clock()
                logging.info("Written elements: %d (%.3fs per element)", counter, took / 100)
        else:
            # print('no backchannel: ' + uttInfo['text'])
            pass

    uttDB.close()
    spkDB.close()


if __name__ == "__main__":
    main()
