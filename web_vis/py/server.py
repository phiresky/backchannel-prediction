import websockets
import asyncio
import sys
from jrtk.preprocessing import NumFeature
from jrtk.features import FeatureType
from typing import Tuple, Dict, Optional, List, Iterator
import json
from collections import OrderedDict
from extract_pfiles_python import readDB, util
from extract_pfiles_python.readDB import DBReader
import math
import os.path
from evaluate import evaluate
from extract_pfiles_python.features import Features


# logging.getLogger().setLevel(logging.DEBUG)
# for handler in logging.getLogger().handlers:
#    handler.setLevel(logging.DEBUG)

def parse_binary_frame_with_metadata(buffer: bytes):
    meta_length = int.from_bytes(buffer[0:4], byteorder='little')
    meta = json.loads(buffer[4:meta_length + 4].decode('ascii'))
    return meta, buffer[4 + meta_length:]


def create_binary_frame_with_metadata(meta_dict: dict, data: bytes):
    meta = json.dumps(meta_dict).encode('ascii')
    if len(meta) % 4 != 0:
        meta += b' ' * (4 - len(meta) % 4)
    meta_length = len(meta).to_bytes(4, byteorder='little')
    return meta_length + meta + data


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


def segsToJSON(reader: DBReader, spkr: str, name: str) -> Dict:
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
    for netversion in sorted(os.listdir(folder)):
        path = join(folder, netversion)
        if not isdir(path) or not isfile(join(path, "train.log")):
            continue
        net_conf_path = join(path, "config.json")
        if isfile(net_conf_path):
            curList = []
            rootList.append({'name': netversion, 'children': curList})
            net_conf = util.load_config(net_conf_path)
            stats = net_conf['train_output']['stats']
            curList.append("best")
            curList.append("best.smooth")
            curList.append("best.smooth.thres")
            for id, info in stats.items():
                curList.append(id)
    return rootList


def get_net_output(convid: str, path: List[str]):
    if path[-1].endswith(".smooth"):
        path[-1] = path[-1][:-len(".smooth")]
        smooth = True
    else:
        smooth = False

    version, id = path
    version_path = os.path.relpath(os.path.realpath(os.path.join("trainNN", "out", version)))
    config_path = os.path.join(version_path, "config.json")
    config = util.load_config(config_path)
    features = Features(config, config_path)
    return features.get_net_output(convid, id, smooth)


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
            meta = {'conversation': conv, 'feature': featname, 'byteOffset': offset}
            await ws.send(create_binary_frame_with_metadata(meta, bytes[offset:min((i + 1) * CHUNKSIZE, len(bytes))]))


async def sendOtherFeature(ws, id, feat):
    await ws.send(json.dumps({
        "id": id,
        "data": feat
    }))


def get_extracted_features(reader: DBReader):
    return OrderedDict([
        ("adc", reader.features.get_adc),
        ("power", reader.features.get_power),
        ("pitch", reader.features.get_pitch),
        ("gaussbc", reader.get_gauss_bc_feature),
        ("combined_feat", reader.features.get_combined_feat)
    ])


def get_features():
    feature_names = list(get_extracted_features(origReader))
    features = [
        dict(name="transcript", children=[dict(name="ISL", children="text,bc".split(",")),
                                          dict(name="Original", children="text,words,bc,is_talking".split(","))]),
        dict(name="extracted", children=feature_names),
        dict(name="NN outputs", children=netsTree),
    ]
    return [
        dict(name="A", children=features),
        dict(name="B", children=features),
        dict(name="microphone", children=[dict(name="extracted", children=feature_names),
                                          dict(name="NN outputs", children=netsTree)])
    ]


def get_larger_threshold_feature(feat: NumFeature, reader: DBReader, name: str, threshold: float, color=[0, 255, 0]):
    ls = []
    for start, end in evaluate.get_larger_threshold(feat, reader, threshold):
        ls.append({'from': start, 'to': end, 'text': 'T', 'color': color})
    return {
        'name': name,
        'typ': 'highlights',
        'data': ls
    }


async def sendFeature(ws, id: str, conv: str, featFull: str):
    if featFull[0] != '/':
        raise Exception("featname must start with /")
    channel, category, *path = featFull.split("/")[1:]
    convid = conv + "-" + channel
    if category == "transcript":
        readerType, featname = path
        reader = islReader if readerType == "ISL" else origReader
        if featname == "bc":
            await sendOtherFeature(ws, id,
                                   {"typ": "highlights", "data": getHighlights(reader, conv, channel)})
        elif featname == "is_talking":
            await sendOtherFeature(ws, id, dict(typ="highlights", data=list(get_talking_feature(reader, convid))))
        elif featname == "text":
            await sendOtherFeature(ws, id, segsToJSON(reader, convid, featFull))
        elif featname == "words":
            await sendOtherFeature(ws, id, segsToJSON(wordsReader, convid, featFull))
    elif category == "NN outputs":
        if path[-1].endswith(".thres"):
            path[-1] = path[-1][:-len(".thres")]
            feature = get_net_output(convid, path)
            await sendOtherFeature(ws, id, get_larger_threshold_feature(feature, origReader, featFull, threshold=0.6))
        elif path[-1].endswith(".bc"):
            path[-1] = path[-1][:-len(".bc")]
            feature = get_net_output(convid, path)
            await sendNumFeature(ws, id, conv, featFull, evaluate.get_bc_audio(feature, origReader, list(
                evaluate.get_bc_samples(origReader, "sw2249-A"))))
        else:
            feature = get_net_output(convid, path)
            await sendNumFeature(ws, id, conv, featFull, feature)
    elif category == "extracted":
        feats = get_extracted_features(origReader)
        featname, = path
        if featname in feats:
            return await sendNumFeature(ws, id, conv, featFull, feats[featname](convid))
        raise Exception("feature not found: {}".format(featFull))
    else:
        raise Exception("unknown category " + category)


def getHighlights(reader: DBReader, conv: str, channel: str):
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
        t = reader.getBCMaxTime(bc)
        highlights.append({'from': t - 0.01, 'to': t + 0.01, 'color': (0, 0, 255), 'text': 'power max'})
        (a, b) = reader.getBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (0, 255, 0), 'text': 'BC'})
        (a, b) = reader.getNonBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (255, 0, 0), 'text': 'NBC'})
    return highlights


def get_talking_feature(reader: DBReader, convid: str):
    for start, end in evaluate.get_talking_segments(reader, convid):
        yield {'from': start, 'to': end, 'text': 'talking', 'color': [255, 255, 0]}


def sanitize_conversation(conv):
    return os.path.basename(conv)


async def handler(websocket, path):
    print("new client connected.")
    while True:
        try:
            data = await websocket.recv()
            if type(data) is bytes:
                meta, data = parse_binary_frame_with_metadata(data)
                print(meta)
            else:
                msg = json.loads(data)
                id = msg['id']
                try:
                    if msg['type'] == "getFeatures":
                        cats = [s.split(" & ") for s in
                                ["/extracted/adc & /transcript/Original/bc",
                                 "/transcript/Original/text", "/extracted/pitch", "/extracted/power"]]
                        conv = sanitize_conversation(msg['conversation'])
                        await websocket.send(json.dumps({"id": id, "data": {
                            'categories': get_features(),
                            'defaults': [["/A" + sub for sub in l] for l in cats] +
                                        [["/B" + sub for sub in l] for l in cats]
                        }}))
                    elif msg['type'] == "getConversations":
                        await websocket.send(json.dumps({"id": id, "data": conversations}))
                    elif msg['type'] == "getFeature":
                        conv = sanitize_conversation(msg['conversation'])
                        await sendFeature(websocket, id, conv, msg['feature'])
                    elif msg['type'] == "echo":
                        await websocket.send(json.dumps({"id": id}))
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
    start_server = websockets.serve(handler, "localhost", 8765)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)

    loop.run_forever()


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = util.load_config(config_path)

    origReader = DBReader(config, config_path)
    config['extract_config']['useWordsTranscript'] = True
    wordsReader = DBReader(config, config_path)
    config['extract_config']['useWordsTranscript'] = False
    config['extract_config']['useOriginalDB'] = False
    islReader = DBReader(config, config_path)
    conversations = readDB.read_conversations(config)
    netsTree = findAllNets()
    start_server()
