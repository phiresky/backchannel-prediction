import websockets, asyncio
import soundfile as sf
import jrtk
from jrtk.preprocessing import NumFeature, FeatureExtractor
from typing import List, Union, Tuple, Dict
from pprint import pprint
import json
import numpy as np

def readAudioFile(featureSet, filename: str, *, dtype='int16', **kwargs) -> Union[NumFeature, List[NumFeature]]:
    """Thin wrapper around the soundfile.read() method. Arguments are passed through, data read is returned as NumFeature.

    For a complete list of arguments and description see http://pysoundfile.readthedocs.org/en/0.8.1/#module-soundfile

    Returns:
        Single NumFeature if the audio file had only 1 channel, otherwise a list of NumFeatures.
    """
    data, samplingRate = sf.read(filename, dtype=dtype, **kwargs)

    if data.ndim == 2:
        # multi channel, each column is a channel
        return [NumFeature(col, featureSet=featureSet, samplingRate=samplingRate / 1000, shift=0) for col in data.T]
    return NumFeature(data, featureSet=featureSet, samplingRate=samplingRate / 1000, shift=0)


def featureToJSON(name: str, feature: NumFeature, range: Union[Tuple[float, float], str]) -> str:
    return json.dumps({
        'name': name,
        'samplingRate': feature.samplingRate,
        'dtype': str(feature.dtype),
        'typ': str(feature.typ),
        'shift': feature.shift,
        'data': feature.tolist(),
        'range': range
    })

def segsToJSON(name: str) -> str:
    return json.dumps({
        'name': name,
        'typ': 'utterances',
        'data': [{**uttDB[utt], 'id': utt} for utt in spkDB[name]['segs'].strip().split(" ")]
    })
db = "../../ears2/db/train/all240302"
conv = 'sw2013'

uttDB = jrtk.base.DBase(baseFilename=db + "-utt", mode="r")
spkDB = jrtk.base.DBase(baseFilename=db + "-spk", mode="r")

featureExtractor = FeatureExtractor(config={'delta': 10, 'base': '../../ears2/earsData'})
featureExtractor.appendStep("../../extract_pfiles_python/featAccess.py")
featureExtractor.appendStep("../../extract_pfiles_python/featDescDelta.py")

async def sendConversation(conv: str, ws):
    features = featureExtractor.eval(None, {'conv': conv, 'from': 0, 'to': 60 * 10})  # type: Dict[str, NumFeature]

    for (name, feat) in sorted(features.items()):
        if name.startswith("feat"): continue
        if 'raw' in name: continue
        # if not name.startswith("adc"): continue
        await ws.send(featureToJSON(name, feat, range=(-2 ** 15, 2 ** 15) if name.startswith("adc") else "normalize"))
        if name == "adca": await ws.send(segsToJSON(conv + '-A'))
        if name == "adcb": await ws.send(segsToJSON(conv + '-B'))

async def handler(websocket, path):
    msg = json.loads(await websocket.recv())
    if msg['type'] == "loadConversation":
        await sendConversation(msg['name'], websocket)
    else:
        raise Exception("Unknown msg " + json.dumps(msg))


start_server = websockets.serve(handler, '0.0.0.0', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
