from extract.readDB import DBReader
from typing import List
import random
from . import evaluate
import numpy as np
import os
import soundfile
import sys
from extract.util import load_config, invert_channel
from extract.readDB import loadDBReader, DBReader, swap_speaker, read_conversations, orig_noise_filter

min_truth_bcs = 3
monologue_length = 15
threshold = 0.62
# blacklist because some have a lot of leaking between channels or bad quality
blacklist = "sw2193".split(",")


def all_mono_segs(reader: DBReader, convs: List[str], min_truth_bcs):
    for conv in convs:
        if conv in blacklist:
            continue
        for channel in ["A", "B"]:
            convchannel = f"{conv}-{channel}"
            bcchannel = f"{conv}-{invert_channel(channel)}"
            for start, end in evaluate.get_monologuing_segments(reader, convchannel, monologue_length):
                bcs = [bc for (bc, bcInfo) in reader.get_backchannels(list(reader.get_utterances(bcchannel))) if
                       float(bcInfo['from']) >= start and float(bcInfo['to']) <= end]
                if len(bcs) >= min_truth_bcs:
                    yield conv, channel, convchannel, start, end


with open("data/backchannels.noncommital") as f:
    noncommitals = set([line.strip() for line in f.readlines() if line[0] != '#'])


def noncommital_bcs(bcInfo):
    # txt = orig_noise_filter(bcInfo['text'])
    # don't filter noise because we don't want e.g. laughter
    return bcInfo['text'].strip() in noncommitals


def write_wavs(reader: DBReader, convs: List[str], count_per_set: int, net_version: str, bc_sample_tracks,
               write_mono=True,
               write_orig=True,
               write_nn=True,
               write_truthrandom=True,
               ):
    all = list(all_mono_segs(reader, convs, min_truth_bcs))
    print(f"found {len(all)} fitting monologue segments of at least {monologue_length}s with ≥ {min_truth_bcs} bcs")
    if count_per_set < len(all):
        random.shuffle(all)
    count = 0
    for conv, channel, convchannel, start, end in all:
        if count == count_per_set:
            return
        bcs = []

        print(f"evaluating conv {convchannel} ({start}s-{end}s)")
        _orig_audio = reader.features.get_adc(convchannel)
        out_dir = os.path.join("evaluate", "out", net_version)
        os.makedirs(out_dir, exist_ok=True)

        start_inx = _orig_audio.time_to_sample_index(start)
        end_inx = _orig_audio.time_to_sample_index(end)
        # minlen = min(_orig_audio.size, nn_bc_audio.size)
        orig_audio = evaluate.normalize_audio(_orig_audio[start_inx:end_inx])

        bcconvchannel = f"{conv}-{invert_channel(channel)}"
        if write_mono:
            soundfile.write(os.path.join(out_dir, f"{conv}{channel} @{start:.1f}s.wav"),
                            orig_audio, 8000)
        if write_nn:
            bc_sampletrack = None
            while len(bcs) < 5:
                bc_sampletrack = random.choice(bc_sample_tracks)
                bcs = list(evaluate.get_bc_samples(reader, noncommital_bcs, bc_sampletrack))
            print(f"with bc samples from {bc_sampletrack}")
            bctrack = bc_sampletrack.replace("-", "")
            smoothed = reader.features.get_net_output(convchannel, "best", smooth=True)

            nn_bc_audio = evaluate.get_bc_audio2(reader, _orig_audio.size, bcs,
                                                 evaluate.predict_bcs(reader, smoothed, threshold=threshold))
            nn_bc_audio = evaluate.normalize_audio(nn_bc_audio[start_inx:end_inx], maxamplitude=0.8)

            soundfile.write(os.path.join(out_dir, f"{conv}{channel} @{start:.1f}s BC=NN-{bctrack}.wav"),
                            np.stack([orig_audio, nn_bc_audio], axis=1), 8000)
        if write_orig:
            truth_bc_audio = reader.features.get_adc(bcconvchannel)
            truth_bc_audio = evaluate.normalize_audio(truth_bc_audio[start_inx:end_inx], maxamplitude=0.8)

            soundfile.write(os.path.join(out_dir, f"{conv}{channel} @{start:.1f}s BC=Truth.wav"),
                            np.stack([orig_audio, truth_bc_audio], axis=1), 8000)
        if write_truthrandom:
            truth_predictor = [reader.getBcRealStartTime(bc) for (bc, bcInfo) in
                               reader.get_backchannels(list(reader.get_utterances(bcconvchannel)))]
            truth_randomized_bc_audio = evaluate.get_bc_audio2(reader, _orig_audio.size, bcs, truth_predictor)
            truth_randomized_bc_audio = evaluate.normalize_audio(truth_randomized_bc_audio[start_inx:end_inx],
                                                                 maxamplitude=0.8)
            soundfile.write(os.path.join(out_dir, f"{conv}{channel} @{start:.1f}s BC=Truth-Randomized.wav"),
                            np.stack([orig_audio, truth_randomized_bc_audio], axis=1), 8000)
        count += 1


good_bc_sample_tracks = "sw2249-A,sw2254-A,sw2258-B,sw2297-A,sw2411-A,sw2432-A,sw2463-A,sw2485-A,sw2603-A,sw2606-B,sw2709-A,sw2735-B,sw2762-A,sw2836-B,sw4193-A".split(
    ",")

if __name__ == '__main__':
    config_path = sys.argv[1]
    args = config_path.split("/")
    version = "None"
    if len(args) == 4:
        _, _, version, _ =  args

    config = load_config(config_path)
    reader = loadDBReader(config_path)
    conversations = read_conversations(config)
    eval_conversations = sorted(conversations['eval'])
    valid_conversations = sorted(conversations['validate'])
    write_wavs(reader, eval_conversations, 1e10, version, good_bc_sample_tracks, write_mono=True, write_nn=False,
               write_orig=False, write_truthrandom=False)
    # write_wavs(reader, eval_conversations, 100000000, version, good_bc_sample_tracks,
    #           )
