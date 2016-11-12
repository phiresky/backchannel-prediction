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

featureExtractor = FeatureExtractor(config={'delta': 10, 'base': '../../ears2/earsData'})
featureExtractor.appendStep("../../extract_pfiles_python/featAccess.py")
featureExtractor.appendStep("../../extract_pfiles_python/featDescDelta.py")
""""[adcA, adcB] = readAudioFile(jrtk.features.FeatureSet(), "../../ears2/earsData/swbLinks/sw2001.wav") # type: NumFeature

def filterPower(self, power: NumFeature) -> NumFeature:
    b = power.max() / 10 ** 4
    power = np.log10(power + b)
    power = power.applyFilter(self.filtr)
    power = power.applyFilter(self.filtr)
    power = power.normalize(min=-0.1, max=0.5)
    return power

vals = {
    'adcA': adcA.substractMean()
'adcB': adcB.substractMean()
'powA': adcA.adc2pow("32ms")
'powB': adcB.adc2pow("32ms")
'powLogA':
}"""
features = featureExtractor.eval(None, {'conv':'sw2001', 'from': 0, 'to': 60 * 10})  # type: Dict[str, NumFeature]

async def handler(websocket, path):
    for (name, feat) in sorted(features.items()):
        if name.startswith("feat"): continue
        # if not name.startswith("adc"): continue
        await websocket.send(featureToJSON(name, feat, range=(-2 ** 15, 2 ** 15) if name.startswith("adc") else "normalize"))
    # await websocket.send(featureToJSON("adc", adcB, range=(-2 ** 15, 2 ** 15)))
    # await websocket.send(featureToJSON("pow", powA, range="normalize"))
    # await websocket.send(featureToJSON("pow2", powB, range="normalize"))
    # name = await websocket.recv()
    # print("< {}".format(name))

    # greeting = "Hello {}!".format(name)
    # await websocket.send(greeting)
    # print("> {}".format(greeting))


start_server = websockets.serve(handler, '0.0.0.0', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
