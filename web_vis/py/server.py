import websockets
import asyncio
import jrtk
from jrtk.preprocessing import NumFeature, FeatureExtractor
from jrtk.features import FeatureType
from typing import Tuple, Dict, Optional
import json
import importlib.util

from trainNN.evaluate import get_network_outputter
from extract_pfiles_python import readDB


def loadModuleFromPath(path: str):
    spec = importlib.util.spec_from_file_location("trainNN", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


current_net = get_network_outputter("extract_pfiles_python/out/v08-pitchnormalization3-context40/train-config.json",
                                    "trainNN/out/v08-pitchnormalization3-1-g03a8cbe-dirty/epoch-003.pkl")


def evaluateNetwork(input: NumFeature) -> NumFeature:
    return NumFeature(current_net(input)[:, [1]])


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


def segsToJSON(name: str) -> Dict:
    return {
        'name': name,
        'typ': 'utterances',
        'data': [{**uttDB[utt], 'id': utt, 'color': (0, 255, 0) if readDB.isBackchannel(uttDB[utt]) else None}
                 for utt in spkDB[name]['segs'].strip().split(" ")]
    }


config = readDB.load_config("extract_pfiles_python/config.json")

uttDB = jrtk.base.DBase(baseFilename=config['databasePrefix'] + "-utt", mode="r")
spkDB = jrtk.base.DBase(baseFilename=config['databasePrefix'] + "-spk", mode="r")
conversations = sorted({spk.split("-")[0] for spk in spkDB})
featureExtractor = FeatureExtractor(config=config)
featureExtractor.appendStep("extract_pfiles_python/featAccess.py")
featureExtractor.appendStep("extract_pfiles_python/featDescDelta.py")


async def sendFeature(ws, name, feat):
    dataextra = feat.typ == FeatureType.SVector
    await ws.send(json.dumps({
        "type": "getFeature",
        "data": featureToJSON(name, feat, range=(-2 ** 15, 2 ** 15) if name.startswith("adc") else None,
                              nodata=dataextra)
    }))
    if dataextra:
        await ws.send(feat.tobytes())


async def sendConversation(conv: str, ws):
    features = featureExtractor.eval(None, {'conv': conv, 'from': 0, 'to': 60 * 100})  # type: Dict[str, NumFeature]

    for name in "adca,pitcha,powera,adcb,pitchb,powerb".split(","):
        feat = features[name]
        await sendFeature(ws, name, feat)
        if name == "adca": await ws.send(json.dumps({"type": "getFeature", "data": segsToJSON(conv + '-A')}))
        if name == "adcb": await ws.send(json.dumps({"type": "getFeature", "data": segsToJSON(conv + '-B')}))
    await ws.send(json.dumps({
        "type": "getFeature", "data": {"name": "adca.bc", "typ": "highlights", "data": getHighlights(conv, "A")}
    }))
    await ws.send(json.dumps({
        "type": "getFeature", "data": {"name": "adcb.bc", "typ": "highlights", "data": getHighlights(conv, "B")}
    }))
    await sendFeature(ws, "NETA", evaluateNetwork(features['feata']))

    await ws.send(json.dumps({"type": "done"}))


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
            if msg['type'] == "loadConversation":
                await sendConversation(msg['name'], websocket)
            elif msg['type'] == "getConversations":
                await websocket.send(json.dumps({"type": "getConversations", "data": conversations}))
            else:
                raise Exception("Unknown msg " + json.dumps(msg))
        except websockets.exceptions.ConnectionClosed as e:
            return


def start_server():
    start_server = websockets.serve(handler, '0.0.0.0', 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


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
