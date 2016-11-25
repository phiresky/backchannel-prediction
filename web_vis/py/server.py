import os

import websockets
import asyncio
import sys
from jrtk.preprocessing import NumFeature
from jrtk.features import FeatureType
from typing import Tuple, Dict, Optional
import json
from collections import OrderedDict
from trainNN.evaluate import get_network_outputter
from extract_pfiles_python import readDB


def evaluateNetwork(net, input: NumFeature) -> NumFeature:
    return NumFeature(net(input)[:, [1]])


def featureToJSON(name: str, feature: NumFeature, range: Optional[Tuple[float, float]], nodata: bool) -> Dict:
    return {
        'name': name,
        'samplingRate': feature.samplingRate,
        'dtype': str(feature.dtype),
        'typ': str(feature.typ),
        'shift': feature.shift,
        'data': None if nodata else feature.tolist(),
        'range': range
    }


def segsToJSON(spkr: str, name: str) -> Dict:
    return {
        'name': name,
        'typ': 'utterances',
        'data': [{**uttDB[utt], 'id': utt, 'color': (0, 255, 0) if readDB.isBackchannel(uttDB[utt]) else None}
                 for utt in spkDB[spkr]['segs'].strip().split(" ")]
    }

def findAllNets():
    import os, glob
    from os.path import join, isdir, isfile, basename
    folder = join("trainNN", "out")
    for netversion in sorted(os.listdir(folder)):
        path = join(folder, netversion)
        if not isdir(path) or not isfile(join(path, "train.log")):
            continue
        net_conf_path = join(path, "config.json")
        if isfile(net_conf_path):
            net_conf = readDB.load_config(net_conf_path)
            stats = net_conf['train_output']['stats']
            bestid = min(stats.keys(), key=lambda k: stats[k]['validation_error'])
            yield join(netversion, "best"), (net_conf_path, bestid)
            for id, info in stats.items():
                yield join(netversion, info['weights']), (net_conf_path, id)


config = readDB.load_config(sys.argv[1])

spkDB, uttDB = readDB.load_db(config['paths'])
conversations = sorted({spk.split("-")[0] for spk in spkDB})
featureExtractor = readDB.load_feature_extractor(config)
nets = OrderedDict(findAllNets())


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

cache = {} # type: Dict[str, Dict[str, NumFeature]]

def getFeatures(conv: str):
    return {
        "defaults":"adca,texta,pitcha,powera,adcb,textb,pitchb,powerb".split(","),
        "optional": list(nets.keys())
    }

def getExtractedFeature(conv: str, feat: str):
    if conv not in cache:
        cache[conv] = featureExtractor.eval(None, {'conv': conv, 'from': 0, 'to': 60 * 100})
    return cache[conv][feat]

async def sendFeature(ws, id: str, conv: str, feat: str):
    if feat == "adca.bc":
        await sendOtherFeature(ws, id, {"name": "adca.bc", "typ": "highlights", "data": getHighlights(conv, "A")})
    elif feat == "adcb.bc":
        await sendOtherFeature(ws, id, {"name": "adcb.bc", "typ": "highlights", "data": getHighlights(conv, "B")})
    elif feat == "NETA":
        await sendNumFeature(ws, id, feat, evaluateNetwork(getExtractedFeature(conv, 'feata')))
    elif feat == "NETB":
        await sendNumFeature(ws, id, feat, evaluateNetwork(getExtractedFeature(conv, 'featb')))
    elif feat == "texta":
        await sendOtherFeature(ws, id, segsToJSON(conv+"-A", feat))
    elif feat == "textb":
        await sendOtherFeature(ws, id, segsToJSON(conv + "-B", feat))
    elif feat in nets:
        config_path, wid = nets[feat]
        net = get_network_outputter(config_path, wid)
        await sendNumFeature(ws, id, feat, evaluateNetwork(net, getExtractedFeature(conv, 'feata')))
    else:
        await sendNumFeature(ws, id, feat, getExtractedFeature(conv, feat))


def getHighlights(conv: str, channel: str):
    if channel == "A":
        bcChannel = "B"
    elif channel == "B":
        bcChannel = "A"
    else:
        raise Exception("unknown channel " + channel)
    bcs = [uttDB[utt]
           for utt in spkDB[conv + "-" + bcChannel]['segs'].strip().split(" ")
           if readDB.isBackchannel(uttDB[utt.strip()])
           ]
    highlights = []
    for bc in bcs:
        (a, b) = readDB.getBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (0, 255, 0), 'text': 'BC'})
        (a, b) = readDB.getNonBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (255, 0, 0), 'text': 'NBC'})
    return highlights


async def handler(websocket, path):
    print("new client connected.")
    while True:
        try:
            msg = json.loads(await websocket.recv())
            id = msg['id']
            if msg['type'] == "getFeatures":
                await websocket.send(json.dumps({"id": id, "data": getFeatures(msg['conversation'])}))
            elif msg['type'] == "getConversations":
                await websocket.send(json.dumps({"id": id, "data": conversations}))
            elif msg['type'] == "getFeature":
                await sendFeature(websocket, id, msg['conversation'], msg['feature'])
            else:
                raise Exception("Unknown msg " + json.dumps(msg))
        except websockets.exceptions.ConnectionClosed as e:
            return


def start_server():
    start_server = websockets.serve(handler, '0.0.0.0', 8765)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)

    loop.run_forever()


def sanity():
    """check if spkDB and uttDB have the same utterances"""
    utts = list()
    for spk in spkDB:
        x = spkDB[spk]
        utts += x['segs'].strip().split(" ")

    utts2 = list()
    x = list(uttDB)
    for utt in uttDB:
        x = uttDB[utt]
        utts2.append(utt)

    print(utts == utts2)


if __name__ == '__main__':
    start_server()
