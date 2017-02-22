from extract.readDB import DBReader
from typing import List, Iterator, Tuple
import random
from . import evaluate
import numpy as np
import os
from os.path import join
import soundfile
import sys
import functools
from extract.util import load_config, invert_channel
from extract.feature import Audio
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


fmt = "wav"
convFmt = "mp3"


def write_convert(fname, data, sample_rate, downmix: bool, convFmt: str):
    import subprocess, os.path
    if downmix:
        data = np.sum(data, axis=1, dtype='int16')
    soundfile.write(fname, data, sample_rate)
    if convFmt is not None:
        outname = os.path.splitext(fname)[0] + "." + convFmt
        subprocess.call(['ffmpeg', '-y', '-loglevel', 'panic', '-i', fname, '-c:a', 'libmp3lame', '-q:a', '3', outname])


def get_bc_audio(reader: DBReader, total_length_audio_index: int, bcs: List[Tuple[str, dict, float, Audio]],
                 predictions: Iterator[float]):
    output_audio = Audio(np.zeros(total_length_audio_index, dtype='int16'), sample_rate_hz=bcs[0][3].sample_rate_hz)

    for peak_s in predictions:
        bc, bcInfo, bc_start_offset, bc_audio = random.choice(bcs)
        audio_len_samples = bc_audio.shape[0]
        # audio_len_s = reader.features.sample_index_to_time(bc_audio, audio_len_samples)
        start_s = peak_s - 0
        start_index = bc_audio.time_to_sample_index(start_s)
        if start_index < 0:
            continue
        if start_index + audio_len_samples > output_audio.shape[0]:
            audio_len_samples = output_audio.shape[0] - start_index
        output_audio[start_index:start_index + audio_len_samples] += bc_audio[0: audio_len_samples]
    return output_audio


want_margin = (-0.2, 0.2)


def write_wavs(reader: DBReader, convs: List[str], count_per_set: int, net_version: str, bc_sample_tracks,
               write_mono=True,
               write_orig=True,
               write_nn=True,
               write_truthrandom=True,
               write_random=True,
               downmix=False
               ):
    all = list(all_mono_segs(reader, convs, min_truth_bcs))
    print(f"found {len(all)} fitting monologue segments of at least {monologue_length}s with ≥ {min_truth_bcs} bcs")
    if count_per_set < len(all):
        random.shuffle(all)
    count = 0
    maxamplitudeOrig = 0.5
    maxamplitudeBC = 0.5
    for conv, channel, convchannel, start, end in all:
        if count == count_per_set:
            return
        out_dir = join("evaluate", "out", net_version)
        os.makedirs(out_dir, exist_ok=True)
        print(f"evaluating conv {convchannel} ({start}s-{end}s)")
        _orig_audio = reader.features.get_adc(convchannel)

        start_inx = _orig_audio.time_to_sample_index(start)
        end_inx = _orig_audio.time_to_sample_index(end)
        # minlen = min(_orig_audio.size, nn_bc_audio.size)
        orig_audio = evaluate.normalize_audio(_orig_audio[start_inx:end_inx], maxamplitude=maxamplitudeOrig)

        bcconvchannel = f"{conv}-{invert_channel(channel)}"
        if write_mono:
            out_dir2 = join(out_dir, "mono")
            os.makedirs(out_dir2, exist_ok=True)
            write_convert(join(out_dir2, f"{conv}{channel} @{start:.2f}s.{fmt}"),
                          orig_audio, 8000, downmix=False, convFmt=None)
        if write_nn or write_truthrandom or write_random:
            bc_sampletrack = None
            bcs = []
            while len(bcs) < 5:
                bc_sampletrack = random.choice(bc_sample_tracks)
                bcs = list(get_boring_bcs(config_path, bc_sampletrack))
            bcs = list(bcs_to_samples(reader, bcs))
            print(f"with bc samples from {bc_sampletrack}")
        if write_nn:
            out_dir2 = join(out_dir, "nn")
            os.makedirs(out_dir2, exist_ok=True)

            bctrack = bc_sampletrack.replace("-", "")
            eval_conf = evaluate.get_best_eval_config(config_path, margin=want_margin)

            predictions = evaluate.get_predictions(config_path, convchannel, eval_conf)
            nn_bc_audio = get_bc_audio(reader, _orig_audio.size, bcs, predictions)
            nn_bc_audio = evaluate.normalize_audio(nn_bc_audio[start_inx:end_inx], maxamplitude=maxamplitudeBC)

            write_convert(join(out_dir2, f"{conv}{channel} @{start:.2f}s BC=NN-{bctrack}.{fmt}"),
                          np.stack([orig_audio, nn_bc_audio], axis=1), 8000, downmix=downmix, convFmt='mp3')
        if write_orig:
            out_dir2 = join(out_dir, "orig")
            os.makedirs(out_dir2, exist_ok=True)
            truth_bc_audio = reader.features.get_adc(bcconvchannel)
            truth_bc_audio = evaluate.normalize_audio(truth_bc_audio[start_inx:end_inx], maxamplitude=maxamplitudeBC)

            write_convert(join(out_dir2, f"{conv}{channel} @{start:.2f}s BC=Truth.{fmt}"),
                          np.stack([orig_audio, truth_bc_audio], axis=1), 8000, downmix=downmix, convFmt='mp3')
        if write_truthrandom:
            out_dir2 = join(out_dir, "truthrandom")
            os.makedirs(out_dir2, exist_ok=True)
            truth_predictor = [reader.getBcRealStartTime(bc) for (bc, bcInfo) in
                               reader.get_backchannels(list(reader.get_utterances(bcconvchannel)))]
            truth_randomized_bc_audio = get_bc_audio(reader, _orig_audio.size, bcs, truth_predictor)
            truth_randomized_bc_audio = evaluate.normalize_audio(truth_randomized_bc_audio[start_inx:end_inx],
                                                                 maxamplitude=maxamplitudeBC)
            write_convert(join(out_dir2, f"{conv}{channel} @{start:.2f}s BC=Truth-Randomized.{fmt}"),
                          np.stack([orig_audio, truth_randomized_bc_audio], axis=1), 8000, downmix=downmix,
                          convFmt='mp3')
        if write_random:
            # selected / relevant from
            # evaluate/out/v050-finunified-16-g1be124b-dirty:lstm-best-features-power,pitch,ffv,word2vec_dim30-slowbatch/results.json
            # so it has same frequency as good nn result
            frequency = 1  # 8256 / 5109
            shuffle_in_talklen = True
            out_dir2 = join(out_dir, "random")
            os.makedirs(out_dir2, exist_ok=True)
            random_predictor = evaluate.random_predictor(reader, convchannel,
                                                         dict(random_baseline=dict(frequency=frequency,
                                                                                   shuffle_in_talklen=shuffle_in_talklen)))
            randomized_bc_audio = get_bc_audio(reader, _orig_audio.size, bcs, random_predictor)
            randomized_bc_audio = evaluate.normalize_audio(randomized_bc_audio[start_inx:end_inx],
                                                           maxamplitude=maxamplitudeBC)
            write_convert(join(out_dir2,
                               f"{conv}{channel} @{start:.2f}s BC=Randomized-{frequency:.1f}x"
                               + f"-{'T' if shuffle_in_talklen else 'A'}.{fmt}"),
                          np.stack([orig_audio, randomized_bc_audio], axis=1), 8000, downmix=downmix,
                          convFmt='mp3')
        count += 1


@functools.lru_cache()
def get_boring_bcs(config_path: str, convid: str):
    reader = loadDBReader(config_path)
    bcs = reader.get_backchannels(list(reader.get_utterances(convid)))
    l = []
    for (bc, bcInfo) in bcs:
        text = bcInfo['text']  # type: str
        if "[laughter" in text or "[noise" in text:
            continue
        filtered = reader.noise_filter(text).lower()
        if reader.bc_to_category[filtered] != 'neutral':
            continue
        l.append((bc, bcInfo))
    return l


def bcs_to_samples(reader: DBReader, bcs):
    for bc, bcInfo in bcs:
        adc = reader.features.get_adc(bcInfo['convid'])
        fromTime = reader.getBcRealStartTime(bc)
        from_index = adc.time_to_sample_index(fromTime)
        to_index = adc.time_to_sample_index(reader.getBcRealFirstEndTime(bc))
        audio = adc[from_index:to_index]
        if not is_pretty_silent(audio):
            yield bc, bcInfo, fromTime, audio


def is_pretty_silent(audio: Audio):
    pow = np.sqrt(sum(audio.astype('float32') ** 2) / len(audio))
    return pow < 500


def output_bc_samples(version_str, convids: List[str]):
    for convid in convids:
        out_dir = join("evaluate", "out", version_str, "BC", convid)
        os.makedirs(out_dir, exist_ok=True)

        for i, (bc, bcInfo, bcStartOffset, audio) in enumerate(
                bcs_to_samples(reader, get_boring_bcs(config_path, convid))):
            text = bcInfo['text']
            pow = np.sqrt(sum(audio.astype('float32') ** 2) / len(audio))
            # print(f"{conv}: {i:03d}: {pow:05.2f}")
            print(f"{convid}: {i}: {text}")
            # audio = evaluate.normalize_audio(audio, maxamplitude=0.9)
            write_convert(join(out_dir, f"{i:03d}.{fmt}"), audio, 8000, downmix=False, convFmt=None)


good_bc_sample_tracks = "sw2249-A,sw2254-A,sw2258-B,sw2297-A,sw2411-A,sw2432-A,sw2485-A,sw2606-B,sw2735-B,sw2762-A,sw4193-A".split(
    ",")
# good_bc_sample_tracks = ["sw2603-A"]

# noise leaking etc.
bad_eval_tracks = ["sw3536-A", "sw2519-B", "sw2854-A", "sw3422-A", "sw2163-B", "sw3384", "sw4028-B", "sw3662", "sw2073",
                   "sw3105", "sw2307", "sw3942", "sw2307", "sw3715", "sw2027", "sw2849", "sw2787", "sw3357", "sw2389"]
# assume problems are symmetric
bad_eval_convos = [track.split("-")[0] for track in bad_eval_tracks]

good_eval_tracks = []

if __name__ == '__main__':
    config_path = sys.argv[1]
    args = config_path.split("/")
    version = "None"
    if len(args) == 4:
        _, _, version, _ = args

    config = load_config(config_path)
    reader = loadDBReader(config_path)
    conversations = read_conversations(config)
    eval_conversations = sorted(conversations['eval'])
    eval_conversations = [convo for convo in eval_conversations if convo not in bad_eval_convos]
    # valid_conversations = sorted(conversations['validate'])
    write_wavs(reader, eval_conversations, 1e10, version, good_bc_sample_tracks, write_mono=True, write_nn=True,
               write_orig=False, write_truthrandom=True, downmix=True)
    output_bc_samples(version, good_bc_sample_tracks)
# write_wavs(reader, eval_conversations, 100000000, version, good_bc_sample_tracks,
#           )
