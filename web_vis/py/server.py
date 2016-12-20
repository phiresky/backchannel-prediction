import websockets
import asyncio
import sys
from jrtk.preprocessing import NumFeature
from jrtk.features import FeatureType
from typing import Tuple, Dict, Optional, List
import json
from collections import OrderedDict
from extract_pfiles_python import readDB
import functools
import math
import os.path
from time import perf_counter
import numpy as np
import random
from extract_pfiles_python.features import Features, NumFeatureCache
import logging

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
            curList.append("best")
            curList.append("best.smooth")
            curList.append("best.smooth.thres")
            for id, info in stats.items():
                curList.append(id)
    return rootList


config_path = sys.argv[1]
config = readDB.load_config(config_path)

origReader = readDB.DBReader(config, config_path)
config['extract_config']['useWordsTranscript'] = True
wordsReader = readDB.DBReader(config, config_path)
config['extract_config']['useWordsTranscript'] = False
config['extract_config']['useOriginalDB'] = False
islReader = readDB.DBReader(config, config_path)
conversations = {name: list(readDB.parse_conversations_file(path)) for name, path in
                 config['paths']['conversations'].items()}
netsTree = findAllNets()


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


def get_extracted_features(reader: readDB.DBReader):
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
                                          dict(name="Original", children="text,words,bc".split(","))]),
        dict(name="extracted", children=feature_names),
        dict(name="NN outputs", children=netsTree),
    ]
    return [
        dict(name="A", children=features),
        dict(name="B", children=features),
        dict(name="microphone", children=[dict(name="extracted", children=feature_names),
                                          dict(name="NN outputs", children=netsTree)])
    ]


def get_net_output(convid: str, path: List[str]):
    if path[-1].endswith(".smooth"):
        path[-1] = path[-1][:-len(".smooth")]
        smooth = True
    else:
        smooth = False

    version, id = path
    version_path = os.path.relpath(os.path.realpath(os.path.join("trainNN", "out", version)))
    config_path = os.path.join(version_path, "config.json")
    config = readDB.load_config(config_path)
    features = Features(config, config_path)
    feature = features.get_net_output(convid, id)

    if smooth:
        import scipy
        res = scipy.ndimage.filters.gaussian_filter1d(feature, 300 / feature.shift, axis=0)
        return NumFeature(res)
    else:
        return feature


def get_larger_threshold(feat: NumFeature, name: str, threshold=0.5, color=[0, 255, 0]):
    begin = None
    ls = []
    for index, [sample] in enumerate(feat):
        if sample >= threshold and begin is None:
            begin = index
        elif sample < threshold and begin is not None:
            ls.append({'from': origReader.features.sample_index_to_time(feat, begin),
                       'to': origReader.features.sample_index_to_time(feat, index), 'text': 'T', 'color': color})
            begin = None

    return {
        'name': name,
        'typ': 'highlights',
        'data': ls
    }


def get_bc_audio(feat: NumFeature):
    sampletrack = "sw4687-B"  # ""sw2807-A"  # "sw3614-A"
    reader = origReader
    sampletrack_audio = reader.features.get_adc(sampletrack)
    bcs = reader.get_backchannels(list(reader.get_utterances(sampletrack)))
    larger_thresholds = get_larger_threshold(feat, "", threshold=0.6)['data']
    total_length_s = reader.features.sample_index_to_time(feat, feat.shape[0])
    total_length_audio_index = reader.features.time_to_sample_index(sampletrack_audio, total_length_s)
    output_audio = NumFeature(np.zeros(total_length_audio_index, dtype='int16'),
                              samplingRate=sampletrack_audio.samplingRate)

    for range in larger_thresholds:
        peak_s = reader.get_max_time(feat, range['from'], range['to']) - reader.BCcenter
        bc_id, bc_info = random.choice(bcs)
        bc_start_time = float(bc_info['from'])
        bc_audio = reader.features.cut_range(sampletrack_audio, bc_start_time, float(bc_info['to']))
        bc_real_start_time = reader.getBcRealStartTime(bc_id)
        bc_start_offset = bc_real_start_time - bc_start_time
        audio_len_samples = bc_audio.shape[0]
        # audio_len_s = reader.features.sample_index_to_time(bc_audio, audio_len_samples)
        start_s = peak_s - bc_start_offset - 0.1
        start_index = reader.features.time_to_sample_index(bc_audio, start_s)
        if start_index < 0:
            continue
        if start_index + audio_len_samples > output_audio.shape[0]:
            audio_len_samples = output_audio.shape[0] - start_index
        output_audio[start_index:start_index + audio_len_samples] += bc_audio[0: audio_len_samples]
    return output_audio


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
        elif featname == "text":
            await sendOtherFeature(ws, id, segsToJSON(reader, convid, featFull))
        elif featname == "words":
            await sendOtherFeature(ws, id, segsToJSON(wordsReader, convid, featFull))
    elif category == "NN outputs":
        if path[-1].endswith(".thres"):
            path[-1] = path[-1][:-len(".thres")]
            feature = get_net_output(convid, path)
            await sendOtherFeature(ws, id, get_larger_threshold(feature, featFull, threshold=0.6))
        elif path[-1].endswith(".bc"):
            path[-1] = path[-1][:-len(".bc")]
            feature = get_net_output(convid, path)
            await sendNumFeature(ws, id, conv, featFull, get_bc_audio(feature))
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
        t = reader.getBCMaxTime(bc)
        highlights.append({'from': t - 0.01, 'to': t + 0.01, 'color': (0, 0, 255), 'text': 'power max'})
        (a, b) = reader.getBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (0, 255, 0), 'text': 'BC'})
        (a, b) = reader.getNonBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (255, 0, 0), 'text': 'NBC'})
    return highlights


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
    start_server()
