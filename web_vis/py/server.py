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


def evaluateNetwork(net, input: NumFeature) -> NumFeature:
    return NumFeature(net(input)[:, [1]])


def featureToJSON(name: str, feature: NumFeature, range: Optional[Tuple[float, float]], nodata: bool) -> Dict:
    return {
        'samplingRate': feature.samplingRate,
        'dtype': str(feature.dtype),
        'typ': str(feature.typ),
        'shift': feature.shift,
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
            accessDict[join(netversion, "best")] = (net_conf_path, bestid)
            curList.append("best")
            for id, info in stats.items():
                accessDict[join(netversion, info['weights'])] = (net_conf_path, id)
                curList.append(info["weights"])
    return accessDict, rootList


config = readDB.load_config(sys.argv[1])

origReader = readDB.DBReader(config)
conversations = sorted({spk.split("-")[0] for spk in origReader.spkDB})
featureExtractor = readDB.load_feature_extractor(config)
netsDict, netsTree = findAllNets()


async def sendNumFeature(ws, id, name, feat):
    dataextra = feat.typ == FeatureType.SVector
    await ws.send(json.dumps({
        "id": id,
        "data": featureToJSON(name, feat, range=(-2 ** 15, 2 ** 15) if name.startswith("adc") else None,
                              nodata=dataextra)
    }))
    if dataextra:
        await ws.send(feat.tobytes())


async def sendOtherFeature(ws, id, feat):
    await ws.send(json.dumps({
        "id": id,
        "data": feat
    }))


def getFeatures(conv: str):
    return [
        dict(name="input", children="adca,texta,bca,adcb,textb,bcb".split(",")),
        dict(name="extracted", children="pitcha,powera,pitchb,powerb,feata,featb".split(",")),
        dict(name="other", children=["islTranscriptA", "islTranscriptB"]),
        dict(name="NN outputs A", children=netsTree),
        dict(name="NN outputs B", children=netsTree),
    ]


@functools.lru_cache(maxsize=1)
def extractFeatures(conv: str):
    return featureExtractor.eval(None, {'conv': conv, 'from': 0, 'to': 60 * 100})


def getExtractedFeature(conv: str, feat: str):
    return extractFeatures(conv)[feat]


async def sendFeature(ws, id: str, conv: str, featFull: str):
    featFull = featFull.lstrip('/')
    category, _, feat = featFull.partition("/")
    if feat == "bca":
        await sendOtherFeature(ws, id,
                               {"name": feat, "typ": "highlights", "data": getHighlights(origReader, conv, "A")})
    elif feat == "bcb":
        await sendOtherFeature(ws, id,
                               {"name": feat, "typ": "highlights", "data": getHighlights(origReader, conv, "B")})
    elif feat == "texta":
        await sendOtherFeature(ws, id, segsToJSON(origReader, conv + "-A", feat))
    elif feat == "textb":
        await sendOtherFeature(ws, id, segsToJSON(origReader, conv + "-B", feat))
    elif feat in netsDict:
        channel = category[-1].lower()
        config_path, wid = netsDict[feat]
        net = get_network_outputter(config_path, wid)
        await sendNumFeature(ws, id, feat, evaluateNetwork(net, getExtractedFeature(conv, 'feat' + channel)))
    else:
        await sendNumFeature(ws, id, feat, getExtractedFeature(conv, feat))


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
                        'defaults': [s.split("&") for s in
                                     ["input/adca&input/bca",
                                      "input/texta", "extracted/pitcha", "extracted/powera",
                                      "input/adcb&input/bcb", "input/textb", "extracted/pitchb", "extracted/powerb"]]
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
