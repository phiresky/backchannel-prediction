import websockets
import asyncio
import sys
from jrtk.preprocessing import NumFeature
from jrtk.features import FeatureType
from typing import Tuple, Dict, Optional, List
import json
from collections import OrderedDict
from trainNN.evaluate import get_network_outputter
from extract_pfiles_python import readDB
import functools
import math
import time


def evaluateNetwork(net, input: NumFeature) -> NumFeature:
    return NumFeature(net(input)[:, [1]])


def featureToJSON(feature: NumFeature, range: Optional[Tuple[float, float]], nodata: bool) -> Dict:
    return {
        'samplingRate': feature.samplingRate,
        'dtype': str(feature.dtype),
        'typ': str(feature.typ),
        'shift': feature.shift,
        'shape': feature.shape,
        'data': None if nodata else feature.tolist(),
        'range': range
    }


def segsToJSON(reader: readDB.DBReader, spkr: str, name: str) -> Dict:
    utts = list(reader.get_utterances(spkr))
    return {
        'name': name,
        'typ': 'utterances',
        'data': [{**uttInfo, 'id': utt,
                  'color': (0, 255, 0) if reader.is_backchannel(uttInfo, index, utts) else None}
                 for index, (utt, uttInfo) in enumerate(utts)]
    }


def findAllNets():
    import os
    from os.path import join, isdir, isfile
    folder = join("trainNN", "out")
    rootList = []
    accessDict = OrderedDict()
    for netversion in sorted(os.listdir(folder)):
        path = join(folder, netversion)
        if not isdir(path) or not isfile(join(path, "train.log")):
            continue
        net_conf_path = join(path, "config.json")
        if isfile(net_conf_path):
            curList = []
            rootList.append({'name': netversion, 'children': curList})
            net_conf = readDB.load_config(net_conf_path)
            stats = net_conf['train_output']['stats']
            bestid = min(stats.keys(), key=lambda k: stats[k]['validation_error'])
            accessDict[(netversion, "best")] = (net_conf_path, bestid)
            curList.append("best")
            for id, info in stats.items():
                accessDict[(netversion, info['weights'])] = (net_conf_path, id)
                curList.append(info["weights"])
    return accessDict, rootList


config = readDB.load_config(sys.argv[1])

origReader = readDB.DBReader(config)
config['extract_config']['useOriginalDB'] = False
islReader = readDB.DBReader(config)
conversations = sorted({spk.split("-")[0] for spk in origReader.spkDB if spk[0:2] == "sw"})
featureExtractor = readDB.load_feature_extractor(config)
netsDict, netsTree = findAllNets()


async def sendNumFeature(ws, id, conv: str, featname: str, feat):
    global lasttime
    dataextra = feat.typ == FeatureType.SVector or feat.typ == FeatureType.FMatrix
    await ws.send(json.dumps({
        "id": id,
        "data": featureToJSON(feat, range=(-2 ** 15, 2 ** 15) if "adc" in featname else None,
                              nodata=dataextra)
    }))
    if dataextra:
        CHUNKSIZE = 200000
        bytes = feat.tobytes()
        # print(feat[0:100])
        # print([x for x in bytes[0:200]])
        for i in range(0, math.ceil(len(bytes) / CHUNKSIZE)):
            offset = i * CHUNKSIZE
            meta = json.dumps({'conversation': conv, 'feature': featname, 'byteOffset': offset}).encode('ascii')
            if len(meta) % 4 != 0:
                meta += b' ' * (4 - len(meta) % 4)
            metaLen = len(meta).to_bytes(4, byteorder='little')
            await ws.send(metaLen + meta + bytes[offset:min((i + 1) * CHUNKSIZE, len(bytes))])


async def sendOtherFeature(ws, id, feat):
    await ws.send(json.dumps({
        "id": id,
        "data": feat
    }))


def getFeatures(conv: str):
    return [
        dict(name="input", children=["adca", "adcb",
                                     dict(name="ISL transcript", children="texta,bca,textb,bcb".split(",")),
                                     dict(name="Original transcript", children="texta,bca,textb,bcb".split(","))
                                     ]),
        dict(name="extracted", children="pitcha,powera,pitchb,powerb,feata,featb".split(",")),
        dict(name="NN outputs A", children=netsTree),
        dict(name="NN outputs B", children=netsTree),
    ]


@functools.lru_cache(maxsize=1)
def extractFeatures(conv: str):
    return featureExtractor.eval(None, {'conv': conv, 'from': 0, 'to': 60 * 100})


def getExtractedFeature(conv: str, feat: str):
    return extractFeatures(conv)[feat]


async def sendFeature(ws, id: str, conv: str, featFull: str):
    if featFull[0] != '/':
        raise Exception("featname must start with /")
    parts = featFull.split("/")[1:]
    if parts[0] == "input":
        if parts[1] in ("adca", "adcb"):
            return await sendNumFeature(ws, id, conv, featFull, getExtractedFeature(conv, parts[1]))
        reader = islReader if parts[1] == "ISL transcript" else origReader
        feat = parts[2]
        if feat == "bca":
            await sendOtherFeature(ws, id,
                                   {"typ": "highlights", "data": getHighlights(reader, conv, "A")})
        elif feat == "bcb":
            await sendOtherFeature(ws, id,
                                   {"typ": "highlights", "data": getHighlights(reader, conv, "B")})
        elif feat == "texta":
            await sendOtherFeature(ws, id, segsToJSON(reader, conv + "-A", feat))
        elif feat == "textb":
            await sendOtherFeature(ws, id, segsToJSON(reader, conv + "-B", feat))
    elif tuple(parts[1:]) in netsDict:
        feat = parts[1:]
        channel = parts[0][-1].lower()
        config_path, wid = netsDict[tuple(feat)]
        net = get_network_outputter(config_path, wid)
        await sendNumFeature(ws, id, conv, featFull, evaluateNetwork(net, getExtractedFeature(conv, 'feat' + channel)))
    else:
        extracted = extractFeatures(conv)
        if parts[1] in extracted:
            return await sendNumFeature(ws, id, conv, featFull, extracted[parts[1]])
        raise Exception("feature not found: {}".format(parts))


def getHighlights(reader: readDB.DBReader, conv: str, channel: str):
    if channel == "A":
        bcChannel = "B"
    elif channel == "B":
        bcChannel = "A"
    else:
        raise Exception("unknown channel " + channel)
    utts = list(reader.get_utterances(conv + "-" + bcChannel))
    bcs = reader.get_backchannels(utts)
    highlights = []
    for bc, bcInfo in bcs:
        (a, b) = reader.getBackchannelTrainingRange(bcInfo)
        highlights.append({'from': a, 'to': b, 'color': (0, 255, 0), 'text': 'BC'})
        (a, b) = reader.getNonBackchannelTrainingRange(bcInfo)
        highlights.append({'from': a, 'to': b, 'color': (255, 0, 0), 'text': 'NBC'})
    return highlights


async def handler(websocket, path):
    print("new client connected.")
    while True:
        try:
            msg = json.loads(await websocket.recv())
            id = msg['id']
            try:
                if msg['type'] == "getFeatures":
                    await websocket.send(json.dumps({"id": id, "data": {
                        'categories': getFeatures(msg['conversation']),
                        'defaults': [s.split(" & ") for s in
                                     ["/input/adca & /input/Original Transcript/bca",
                                      "/input/Original Transcript/texta", "/extracted/pitcha", "/extracted/powera",
                                      "/input/adcb & /input/Original Transcript/bcb",
                                      "/input/Original Transcript/textb",
                                      "/extracted/pitchb", "/extracted/powerb"
                                      ]]
                    }}))
                elif msg['type'] == "getConversations":
                    await websocket.send(json.dumps({"id": id, "data": conversations}))
                elif msg['type'] == "getFeature":
                    await sendFeature(websocket, id, msg['conversation'], msg['feature'])
                else:
                    raise Exception("Unknown msg " + json.dumps(msg))
            except Exception as e:
                if isinstance(e, websockets.exceptions.ConnectionClosed): raise e
                import traceback
                await websocket.send(json.dumps({"id": id, "error": traceback.format_exc()}))
                print(traceback.format_exc())

        except websockets.exceptions.ConnectionClosed as e:
            return


def start_server():
    start_server = websockets.serve(handler, '0.0.0.0', 8765)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)

    loop.run_forever()


if __name__ == '__main__':
    start_server()
