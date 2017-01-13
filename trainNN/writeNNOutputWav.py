import sys
import random
import soundfile
import os
import numpy as np
from jrtk.preprocessing import NumFeature
from web_vis.py.server import get_bc_audio, get_larger_threshold, get_net_output, read_conversations, get_bc_samples, \
    get_talking_segments
from typing import List
from extract_pfiles_python.readDB import load_config, DBReader

from extract_pfiles_python.features import Features

version = sys.argv[1]

config_path = os.path.join("trainNN", "out", version, "config.json")

count_per_set = int(sys.argv[2])

config = load_config(config_path)

conversations = read_conversations(config)
allconversations = [convid for convlist in conversations.values() for conv in convlist for convid in
                    [conv + "-" + channel for channel in ["A", "B"]]]

reader = DBReader(config, config_path)


def normalize_audio(sampletrack_audio):
    max_amplitude = max(abs(sampletrack_audio.max()), abs(sampletrack_audio.min()))
    multi = 32767 / max_amplitude
    return (sampletrack_audio * multi).astype("int16")


def output_convs(convs: List[str]):
    random.shuffle(convs)
    for conv in convs[0:count_per_set]:
        channel = random.choice(["A", "B"])
        convchannel = "{}-{}".format(conv, channel)
        smoothed = reader.features.get_net_output(convchannel, "best", smooth=True)
        bcs = []
        bc_sampletrack = None
        while len(bcs) < 5:
            bc_sampletrack = random.choice(allconversations)
            bcs = list(get_bc_samples(reader, bc_sampletrack))
        bc_audio = get_bc_audio(smoothed, reader, bcs)
        print("doing conv {} with bc samples from {}".format(convchannel, bc_sampletrack))
        _orig_audio = reader.features.get_adc(convchannel)
        out_dir = os.path.join("evaluate", version)
        os.makedirs(out_dir, exist_ok=True)
        minlen = min(_orig_audio.size, bc_audio.size)
        orig_audio = normalize_audio(_orig_audio[0:minlen].reshape((minlen, 1)))
        bc_audio = normalize_audio(bc_audio[0:minlen].reshape((minlen, 1)))
        audio = np.append(orig_audio, bc_audio, axis=1)
        # audio_cut = NumFeature(np.array([], dtype="float32").reshape((0, 2)))
        for start, end in get_talking_segments(reader, convchannel):
            end += 1
            length = end - start
            if length < 15:
                continue
            start_inx = reader.features.time_to_sample_index(_orig_audio, start)
            end_inx = reader.features.time_to_sample_index(_orig_audio, end)
            soundfile.write(os.path.join(out_dir, "SPK={}{} @{:.1f}s BC={}.wav".format(conv, channel, start,
                                                                                       bc_sampletrack.replace("-",
                                                                                                              ""))),
                            audio[start_inx:end_inx], 8000)
            # audio_cut = np.append(audio_cut, audio[start_inx:end_inx])

            # soundfile.write(os.path.join(out_dir, "{}--{}.wav".format(convchannel, bc_sampletrack)), audio_cut, 8000)


def output_bc_samples(convs: List[str]):
    for conv in convs:
        adc = reader.features.get_adc(conv)
        bcs = reader.get_backchannels(list(reader.get_utterances(conv)))

        def generator():
            for (bc, bcInfo) in bcs:
                from_index = reader.features.time_to_sample_index(adc, float(bcInfo['from']))
                to_index = reader.features.time_to_sample_index(adc, float(bcInfo['to']))
                yield adc[from_index:to_index]

        audio_cut = np.concatenate(generator())
        out_dir = os.path.join("tmp")
        os.makedirs(out_dir, exist_ok=True)
        soundfile.write(os.path.join(out_dir, "{}.wav".format(conv)), audio_cut, 8000)


# output_convs(conversations['eval'])
# output_bc_samples(allconversations)
print("\n".join(allconversations))
