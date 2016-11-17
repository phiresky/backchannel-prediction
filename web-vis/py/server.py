import websockets, asyncio
import soundfile as sf
import jrtk
from jrtk.preprocessing import NumFeature, FeatureExtractor
from typing import List, Union, Tuple, Dict, Optional
from pprint import pprint
import json
import numpy as np

import importlib.util

readDBspec = importlib.util.spec_from_file_location("readDB", "../../extract_pfiles_python/readDB.py")
readDB = importlib.util.module_from_spec(readDBspec)
readDBspec.loader.exec_module(readDB)

lbox = jrtk.hmm.Labelbox()
lbox.load("../../ears2/earsData/2/sw2582-B.lbox.gz")

def featureToJSON(name: str, feature: NumFeature, range: Optional[Tuple[float, float]]) -> Dict:
    return {
        'name': name,
        'samplingRate': feature.samplingRate,
        'dtype': str(feature.dtype),
        'typ': str(feature.typ),
        'shift': feature.shift,
        'data': feature.tolist(),
        'range': range
    }


def segsToJSON(name: str) -> Dict:
    return {
        'name': name,
        'typ': 'utterances',
        'data': [{**uttDB[utt], 'id': utt, 'color': (0, 255, 0) if readDB.isBackchannel(uttDB[utt]) else None}
                 for utt in spkDB[name]['segs'].strip().split(" ")]
    }


db = "../../ears2/db/train/all240302"

uttDB = jrtk.base.DBase(baseFilename=db + "-utt", mode="r")
spkDB = jrtk.base.DBase(baseFilename=db + "-spk", mode="r")
conversations = sorted({spk.split("-")[0] for spk in spkDB})
featureExtractor = FeatureExtractor(config={'delta': 10, 'base': '../../ears2/earsData'})
featureExtractor.appendStep("../../extract_pfiles_python/featAccess.py")
featureExtractor.appendStep("../../extract_pfiles_python/featDescDelta.py")


async def sendConversation(conv: str, ws):
    features = featureExtractor.eval(None, {'conv': conv, 'from': 0, 'to': 60*100})  # type: Dict[str, NumFeature]

    for name in "adca,pitcha,powera,adcb,pitchb,powerb".split(","):
        feat = features[name]
        if name.startswith("feat"): continue
        if 'raw' in name: continue
        # if not name.startswith("adc"): continue
        await ws.send(json.dumps({
            "type": "getFeature",
            "data": featureToJSON(name, feat, range=(-2 ** 15, 2 ** 15) if name.startswith("adc") else None)
        }))
        if name == "adca": await ws.send(json.dumps({"type": "getFeature", "data": segsToJSON(conv + '-A')}))
        if name == "adcb": await ws.send(json.dumps({"type": "getFeature", "data": segsToJSON(conv + '-B')}))
    await ws.send(json.dumps({
        "type": "getHighlights", "data": {"feature": "adca", "highlights": getHighlights(conv, "A")}
    }))
    await ws.send(json.dumps({
        "type": "getHighlights", "data": {"feature": "adcb", "highlights": getHighlights(conv, "B")}
    }))


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
        highlights.append({'from': a, 'to': b, 'color': (0,255,0)})
        (a, b) = readDB.getNonBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (255,0,0)})
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


start_server = websockets.serve(handler, '0.0.0.0', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
