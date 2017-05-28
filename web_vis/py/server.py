import websockets
import asyncio
import sys
from typing import Tuple, Dict, Optional, List, Iterator
import json
from collections import OrderedDict
from extract import readDB, util, features
from extract.readDB import DBReader
import trainNN
import math
import os.path
from evaluate import evaluate, write_wavs
from extract.features import Features
from extract.feature import Feature, Audio
import numpy as np


# logging.getLogger().setLevel(logging.DEBUG)
# for handler in logging.getLogger().handlers:
#    handler.setLevel(logging.DEBUG)


def create_binary_frame_with_metadata(meta_dict: dict, data: bytes):
    meta = json.dumps(meta_dict).encode('ascii')
    if len(meta) % 4 != 0:
        meta += b' ' * (4 - len(meta) % 4)
    meta_length = len(meta).to_bytes(4, byteorder='little')
    return meta_length + meta + data


def featureToJSON(feature: Feature, range: Optional[Tuple[float, float]], nodata: bool) -> Dict:
    if isinstance(feature, Audio):
        return {
            'samplingRate': feature.sample_rate_hz / 1000,
            'dtype': str(feature.dtype),
            'typ': 'FeatureType.SVector',
            'shape': feature.shape,
            'data': None if nodata else feature.tolist(),
            'range': range
        }
    elif isinstance(feature, Feature):
        return {
            'dtype': str(feature.dtype),
            'typ': 'FeatureType.FMatrix',
            'shift': feature.frame_shift_ms,
            'shape': feature.shape,
            'data': None if nodata else feature.tolist(),
            'range': range
        }


def segsToJSON(reader: DBReader, spkr: str, name: str, words=False) -> Dict:
    id = 0
    utts = []
    if words:
        for utt, uttInfo in reader.uttDB.get_utterances(spkr):
            for word in reader.uttDB.get_words_for_utterance(utt, uttInfo):
                id += 1
                utts.append((id, word))
    else:
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


want_margin = (-0.2, 0.2)  # (-0.1, 0.5)


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
    eval_conf = evaluate.get_best_eval_config(config_path, margin=want_margin)
    if smooth:
        return config_path, eval_conf, features.smooth(convid, id, eval_conf['smoother'])
    else:
        return config_path, eval_conf, features.get_multidim_net_output(convid, id)


async def sendNumFeature(ws, id, conv: str, featname: str, feat):
    global lasttime
    dataextra = True
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
    return OrderedDict([(name, getattr(reader.features, f"get_{name}")) for name in
                        "adc,power,raw_power,ffv,mfcc,pitch,word2vec_v1,combined_feature".split(",")])


def get_features():
    feature_names = list(get_extracted_features(origReader))
    features = [
        dict(name="transcript", children="text,words,bc,is_talking,is_silent,is_monologuing".split(",")),
        dict(name="extracted", children=feature_names),
        dict(name="NN outputs", children=netsTree),
    ]
    return [
        dict(name="A", children=features),
        dict(name="B", children=features),
        dict(name="microphone", children=[dict(name="extracted", children=feature_names),
                                          dict(name="NN outputs", children=netsTree)])
    ]


def get_larger_threshold_feature(feat: Feature, reader: DBReader, name: str, threshold: float, color=[0, 255, 0]):
    ls = []
    for start, end in evaluate.get_larger_threshold(feat, reader, threshold):
        ls.append({'from': start, 'to': end, 'text': 'Threshold', 'color': color})
    return {
        'name': name,
        'typ': 'highlights',
        'data': ls
    }


def maybe_onedim(fes):
    single_out_dim = True
    if single_out_dim:
        return 1 - fes[:, [0]]
    else:
        return fes


async def sendFeature(ws, id: str, conv: str, featFull: str, micro):
    if featFull[0] != '/':
        raise Exception("featname must start with /")
    channel, category, *path = featFull.split("/")[1:]
    convid = conv + "-" + channel
    if category == "transcript":
        (featname,) = path
        reader = origReader
        if featname == "bc":
            await sendOtherFeature(ws, id,
                                   {"typ": "highlights", "data": getHighlights(reader, conv, channel)})
        elif featname == "is_talking":
            await sendOtherFeature(ws, id, dict(typ="highlights", data=list(get_talking_feature(reader, convid))))
        elif featname == "is_silent":
            await sendOtherFeature(ws, id, dict(typ="highlights", data=list(get_silent_feature(reader, convid))))
        elif featname == "is_monologuing":
            await sendOtherFeature(ws, id, dict(typ="highlights", data=list(get_monologuing_feature(reader, convid))))
        elif featname == "text":
            await sendOtherFeature(ws, id, segsToJSON(reader, convid, featFull))
        elif featname == "words":
            await sendOtherFeature(ws, id, segsToJSON(origReader, convid, featFull, words=True))
        else:
            raise Exception("unknown trans feature: " + featname)
    elif category == "NN outputs":
        if path[-1].endswith(".thres"):
            path[-1] = path[-1][:-len(".thres")]
            config_path, eval_config, feature = get_net_output(convid, path)
            feature = maybe_onedim(feature)
            onedim = feature if feature.shape[1] == 1 else 1 - feature[:, [0]]
            await sendOtherFeature(ws, id, get_larger_threshold_feature(onedim, origReader, featFull,
                                                                        threshold=eval_config['threshold']))
        elif path[-1].endswith(".bc"):
            path[-1] = path[-1][:-len(".bc")]
            config_path, eval_config, feature = get_net_output(convid, path)
            feature = maybe_onedim(feature)
            onedim = feature if feature.shape[1] == 1 else 1 - feature[:, [0]]
            predictions = evaluate.get_predictions(config_path, convid, eval_config)
            _orig_audio = origReader.features.get_adc(convid)
            import random
            st = random.choice(write_wavs.good_bc_sample_tracks)
            print(f"bcs from {st}")
            bc_audio = write_wavs.get_bc_audio(origReader, _orig_audio.size, list(
                write_wavs.bcs_to_samples(
                    readDB.loadDBReader(config_path),
                    write_wavs.get_boring_bcs(config_path, st))),
                                               predictions)
            await sendNumFeature(ws, id, conv, featFull, bc_audio)
        else:
            _, _, feature = get_net_output(convid, path)
            feature = maybe_onedim(feature)
            await sendNumFeature(ws, id, conv, featFull, feature)
    elif category == "extracted":
        featname, = path
        if channel == "microphone":
            feats = micro.features
            if featname in feats:
                return await sendNumFeature(ws, id, conv, featFull, feats[featname])
        else:
            feats = get_extracted_features(origReader)
            if featname in feats:
                featout = feats[featname](convid)
                # if featname == "pitch":
                #    featout = -featout
                return await sendNumFeature(ws, id, conv, featFull, featout)

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
        t = reader.getBcRealStartTime(bc)
        highlights.append({'from': t - 0.01, 'to': t + 0.01, 'color': (0, 0, 255), 'text': 'real BC start'})
        (a, b) = reader.getBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (0, 255, 0), 'text': 'BC'})
        (a, b) = reader.getNonBackchannelTrainingRange(bc)
        highlights.append({'from': a, 'to': b, 'color': (255, 0, 0), 'text': 'NBC'})
    return highlights


def get_talking_feature(reader: DBReader, convid: str):
    for start, end in evaluate.get_talking_segments(reader, convid, invert=False):
        yield {'from': start, 'to': end, 'text': 'talking', 'color': [255, 255, 0]}


def get_silent_feature(reader: DBReader, convid: str):
    for start, end in evaluate.get_talking_segments(reader, convid, invert=True):
        yield {'from': start, 'to': end, 'text': 'listening', 'color': [176, 224, 230]}


def get_monologuing_feature(reader: DBReader, convid: str):
    for start, end in evaluate.get_monologuing_segments(reader, convid):
        yield {'from': start, 'to': end, 'text': 'monologuing', 'color': [255, 165, 0]}


def sanitize_conversation(conv):
    if conv is None:
        return ""
    return os.path.basename(conv)


class MicrophoneHandler:
    client_sample_rate = 8000
    client_dtype = 'float32'
    buffer_length_s = 60
    client_microphone = Audio(np.zeros(client_sample_rate * buffer_length_s, dtype='int16'), client_sample_rate)
    previous_end_offset_samples = 0
    frame_window_ms = 32  # config['extract_config']['sample_window_ms']
    window_shift_ms = 10
    do_nn = "trainNN/out/v048-finunified-15-g92ee0a9-dirty:lstm-best-features-power,pitch,ffv/config.json"
    dimensions = {
        'ffv': 7,
        'raw_power': 1,
        'pitch': 1
    }
    featurenames = []

    def __init__(self, reader: DBReader, client_socket):
        self.reader = reader
        self.client_socket = client_socket
        for featname in self.reader.config['extract_config']['input_features']:
            pref, featnamepart = featname[0:4], featname[4:]
            if pref != "get_":
                raise Exception(f"invalid featname {featname}")
            self.featurenames.append(featnamepart)
        self.features = {featurename: Feature(
            np.zeros((round(1000 * self.buffer_length_s / self.window_shift_ms), self.dimensions[featurename]),
                     dtype=np.float32),
            infofrom=dict(
                frame_window_ms=self.frame_window_ms,
                frame_shift_ms=self.window_shift_ms,
                initial_frame_offset_ms=0
            ))
            for featurename in self.featurenames}
        if self.do_nn is not None:
            self.nn_config = util.load_config(self.do_nn)
            eval_config = evaluate.get_best_eval_config(self.do_nn)
            epoch, _ = trainNN.evaluate.get_best_epoch(self.nn_config)
            layers, self.nn_fn = trainNN.evaluate.get_network_outputter(self.do_nn, epoch, batch_size=None)
            self.context_frames = self.nn_config['train_config']['context_frames']
            self.context_stride = self.nn_config['train_config']['context_stride']
            self.context_range = self.context_frames * self.context_stride

    def nn_eval(self, input):
        # adapted from extract.features.pure_get_multidim_net_output
        fn = self.nn_fn
        config = self.nn_config
        total_frames = input.shape[0]
        if 0 + self.context_range >= total_frames:
            raise Exception(f"{self.context_range} >= {total_frames} (not enough context given)")

        def stacker():
            for frame in range(0 + self.context_range, total_frames):
                yield input[range(frame - self.context_range, frame, self.context_stride)]

        inp = np.stack(stacker())
        output = features.batched_eval(inp, fn, out_dim=config['train_config']['num_labels'])
        return Feature(output, infofrom=input)

    def handle(self, data: bytes):
        meta, data = self.parse_binary_frame_with_metadata(data)
        data = (data * (1 << 15)).astype('int16')
        if meta['feature'] != '/microphone/extracted/adc':
            print("unknown client feature: " + meta['feature'])
            return
        offset = meta['byteOffset'] // 4
        end_offset = offset + data.shape[0]
        if end_offset > self.client_microphone.shape[0]:
            print("warning: offset > buffer length")
        else:
            print("setting audio", offset, end_offset)
            self.client_microphone[offset:end_offset] = data
            self.microphone_data_changed(end_offset)

    def parse_binary_frame_with_metadata(self, buffer: bytes):
        meta_length = int.from_bytes(buffer[0:4], byteorder='little')
        meta = json.loads(buffer[4:meta_length + 4].decode('ascii'))
        return meta, np.frombuffer(buffer, self.client_dtype, offset=4 + meta_length)

    def microphone_data_changed(self, new_end_offset):
        if (new_end_offset - self.previous_end_offset_samples) / self.client_sample_rate * 1000 >= self.window_shift_ms:
            frame_shift_samples = round(self.window_shift_ms * self.client_sample_rate / 1000)
            frame_window_samples = round(self.frame_window_ms * self.client_sample_rate / 1000)
            exact_offset_in_frames = self.previous_end_offset_samples // frame_shift_samples
            exact_offset = exact_offset_in_frames * frame_shift_samples
            exact_end_offset = ((
                                    new_end_offset - exact_offset - frame_window_samples) // frame_shift_samples) * frame_shift_samples + exact_offset + frame_window_samples
            expected_frames = ((exact_end_offset - frame_window_samples) - exact_offset) // frame_shift_samples
            audio_cut = self.client_microphone[exact_offset:exact_end_offset - 1]
            if expected_frames == 0:
                return
            gottenfeats = []
            for featname in self.featurenames:
                trafo = getattr(features, featname + "_transform")
                feat = trafo(audio_cut, self.frame_window_ms)
                if expected_frames != feat.shape[0]:
                    print(
                        f"expected {expected_frames}, got {feat.shape[0]} ({exact_offset}-{exact_end_offset} ({self.previous_end_offset_samples}-{new_end_offset})) s={frame_shift_samples} w={frame_window_samples}")
                self.features[featname][exact_offset_in_frames: exact_offset_in_frames + expected_frames] = feat
                gottenfeats.append(feat)
                self.emit_change(featname, exact_offset_in_frames, feat)
            if self.do_nn is not None and exact_offset_in_frames - self.context_range >= 0:
                fes = [self.features[f][exact_offset_in_frames - self.context_range:exact_offset_in_frames + expected_frames] for f in
                     self.featurenames]
                combined_input = Feature(np.concatenate(fes, axis=1), infofrom=self.features[self.featurenames[0]])
                output = self.nn_eval(combined_input)
                print(output)
            self.previous_end_offset_samples = exact_offset + frame_shift_samples * expected_frames

    def emit_change(self, featname, frame_offset: int, feat):
        featbytes = feat.tobytes()
        meta = {'conversation': "sw2807", 'feature': f"/microphone/extracted/{featname}",
                'byteOffset': frame_offset * 4 * feat.shape[1]}
        asyncio.ensure_future(self.client_socket.send(create_binary_frame_with_metadata(meta, featbytes)))


async def handler(websocket, path):
    print("new client connected.")
    microphone_handler = MicrophoneHandler(origReader, websocket)
    while True:
        try:
            data = await websocket.recv()
            if type(data) is bytes:
                microphone_handler.handle(data)
            else:
                msg = json.loads(data)
                id = msg['id']
                try:
                    if msg['type'] == "getFeatures":
                        cats = [s.split(" & ") for s in
                                [
                                    "/extracted/mfcc"]]
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
                        await sendFeature(websocket, id, conv, msg['feature'], microphone_handler)
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
    start_server = websockets.serve(handler, "0.0.0.0", 8765)
    print("server started")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)

    loop.run_forever()


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = util.load_config(config_path)

    origReader = DBReader(config_path, originalDb=True)

    conversations = readDB.read_conversations(config)
    netsTree = findAllNets()
    start_server()
