import json
import sys
import random
import soundfile
import os
import numpy as np
from jrtk.preprocessing import NumFeature
from typing import List, Tuple, Iterator
from extract_pfiles_python.readDB import loadDBReader, DBReader, swap_speaker, read_conversations
from extract_pfiles_python.util import load_config
from tqdm import tqdm
import functools
import trainNN.evaluate
from itertools import product

os.environ['JOBLIB_START_METHOD'] = 'forkserver'
from joblib import Parallel, delayed


def get_talking_segments(reader: DBReader, convid: str, min_talk_len=None) -> Iterator[Tuple[float, float]]:
    talk_start = 0
    talking = False
    utts = list(reader.get_utterances(convid))
    for index, (utt, uttInfo) in enumerate(utts):
        is_bc = reader.is_backchannel(uttInfo, index, utts)
        is_empty = len(reader.noise_filter(uttInfo['text'])) == 0
        if is_bc or is_empty:
            if talking:
                talking = False
                talk_end = float(uttInfo['from'])
                if min_talk_len is None or talk_end - talk_start >= min_talk_len:
                    yield talk_start, talk_end
        else:
            if not talking:
                talking = True
                talk_start = float(uttInfo['from'])


def get_bc_samples(reader: DBReader, sampletrack="sw4687-B"):
    sampletrack_audio = reader.features.get_adc(sampletrack)
    bcs = reader.get_backchannels(list(reader.get_utterances(sampletrack)))
    for bc_id, bc_info in bcs:
        bc_start_time = float(bc_info['from'])
        bc_audio = reader.features.cut_range(sampletrack_audio, bc_start_time, float(bc_info['to']))
        bc_real_start_time = reader.getBcRealStartTime(bc_id)
        bc_start_offset = bc_real_start_time - bc_start_time
        yield bc_start_offset, bc_audio


# the word-aligned beginning of the bc is predicted
def predict_bcs(reader: DBReader, smoothed_net_output: NumFeature, threshold: float):
    for start, end in get_larger_threshold(smoothed_net_output, reader, threshold):
        peak_prediction_s = reader.get_max_time(smoothed_net_output, start, end) - ((reader.BCend + reader.BCbegin) / 2)
        yield peak_prediction_s


def get_bc_audio(smoothed_net_output: NumFeature, reader: DBReader,
                 bcs: List[Tuple[int, NumFeature]]):
    total_length_s = reader.features.sample_index_to_time(smoothed_net_output, smoothed_net_output.shape[0])
    total_length_audio_index = reader.features.time_to_sample_index(bcs[0][1], total_length_s)
    output_audio = NumFeature(np.zeros(total_length_audio_index, dtype='int16'),
                              samplingRate=bcs[0][1].samplingRate)

    for peak_s in predict_bcs(reader, smoothed_net_output, threshold=0.6):
        bc_start_offset, bc_audio = random.choice(bcs)
        audio_len_samples = bc_audio.shape[0]
        # audio_len_s = reader.features.sample_index_to_time(bc_audio, audio_len_samples)
        start_s = peak_s - bc_start_offset
        start_index = reader.features.time_to_sample_index(bc_audio, start_s)
        if start_index < 0:
            continue
        if start_index + audio_len_samples > output_audio.shape[0]:
            audio_len_samples = output_audio.shape[0] - start_index
        output_audio[start_index:start_index + audio_len_samples] += bc_audio[0: audio_len_samples]
    return output_audio


def get_first_true_inx(bool_arr, start_inx: int):
    inx = np.argmax(bool_arr[start_inx:]) + start_inx
    if not bool_arr[inx]:
        return None
    return inx


def get_first_false_inx(bool_arr, start_inx: int):
    inx = np.argmin(bool_arr[start_inx:]) + start_inx
    if bool_arr[inx]:
        return None
    return inx


def get_larger_threshold(feat: NumFeature, reader: DBReader, threshold=0.5):
    begin = None
    larger_threshold = feat.reshape((feat.size,)) >= threshold
    inx = 0
    while True:
        start = get_first_true_inx(larger_threshold, inx)
        if start is None:
            return
        end = get_first_false_inx(larger_threshold, start)
        if end is None:
            return
        yield reader.features.sample_index_to_time(feat, start), reader.features.sample_index_to_time(feat, end)
        inx = end + 1


def normalize_audio(sampletrack_audio):
    max_amplitude = max(abs(sampletrack_audio.max()), abs(sampletrack_audio.min()))
    multi = 32767 / max_amplitude
    return (sampletrack_audio * multi).astype("int16")


# is this nearer than that from other?
def nearer(this: float, that: float, other):
    return abs(this - other) < abs(that - other)


def bc_is_within_margin_of_error(predicted: float, correct: float, margin: Tuple[float, float]):
    return correct + margin[0] <= predicted <= correct + margin[1]


def filter_ranges(numbers: List[float], ranges: List[Tuple[float, float]]):
    inx = 0
    if len(numbers) == 0:
        return
    for start, end in ranges:
        while numbers[inx] < start:
            inx += 1
            if inx >= len(numbers):
                return
        while numbers[inx] <= end:
            yield numbers[inx]
            inx += 1
            if inx >= len(numbers):
                return


def evaluate_conv(config_path: str, convid: str, config: dict):
    reader = loadDBReader(config_path)
    bc_convid = swap_speaker(convid)
    correct_bcs = [reader.getBcRealStartTime(utt) for utt, uttInfo in
                   reader.get_backchannels(list(reader.get_utterances(bc_convid)))]

    net_output = reader.features.get_net_output(convid, config["epoch"], smooth=True)
    predicted_bcs = list(predict_bcs(reader, net_output, threshold=config['threshold']))
    predicted_count = len(predicted_bcs)
    predicted_inx = 0
    if predicted_count > 0:
        for correct_bc in correct_bcs:
            while predicted_inx < predicted_count - 1 and nearer(predicted_bcs[predicted_inx + 1],
                                                                 predicted_bcs[predicted_inx], correct_bc):
                predicted_inx += 1
            if bc_is_within_margin_of_error(predicted_bcs[predicted_inx], correct_bc, config['margin_of_error']):
                predicted_bcs[predicted_inx] = correct_bc

    if config['min_talk_len'] is not None:
        segs = list(get_talking_segments(reader, convid, min_talk_len=config['min_talk_len']))
        predicted_bcs = filter_ranges(predicted_bcs, segs)
        correct_bcs = filter_ranges(correct_bcs, segs)
    # https://www.wikiwand.com/en/Precision_and_recall
    selected = set(predicted_bcs)
    relevant = set(correct_bcs)
    true_positives = selected & relevant
    false_positives = selected - relevant
    false_negatives = relevant - selected

    return convid, dict(selected=len(selected), relevant=len(relevant), true_positives=len(true_positives),
                        false_positives=len(false_positives), false_negatives=len(false_negatives))


def precision_recall(stats: dict):
    if stats['true_positives'] == 0:
        # http://stats.stackexchange.com/a/16242
        recall = 1
        precision = 0 if stats['false_positives'] > 0 else 1
    else:
        precision = stats['true_positives'] / stats['selected']
        recall = stats['true_positives'] / stats['relevant']

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return dict(precision=precision, recall=recall, f1_score=f1_score)


def interesting_configs():
    # http://eprints.eemcs.utwente.nl/22780/01/dekok_2012_surveyonevaluation.pdf
    interesting_margins = [(-0.1, 0.5), (-0.5, 0.5), (0, 1), (-1, 0), (-0.2, 0.2)]
    interesting_thresholds = [0.5, 0.6, 0.7]
    interesting_talk_lens = [None, 5, 10]
    for margin, threshold, talk_len in product(interesting_margins, interesting_thresholds, interesting_talk_lens):
        yield dict(margin_of_error=margin, threshold=threshold, epoch="best", min_talk_len=talk_len)


def evaluate_convs(parallel, config_path: str, convs: List[str], eval_config: dict):
    totals = {}
    results = {}
    if "weights_file" not in eval_config and eval_config["epoch"] == "best":
        _, eval_config["weights_file"] = trainNN.evaluate.get_best_epoch(load_config(config_path))
    convids = ["{}-{}".format(conv, channel) for conv in convs for channel in ["A", "B"]]
    for convid, result in parallel(
            tqdm([delayed(evaluate_conv)(config_path, convid, eval_config) for convid in convids])):
        results[convid] = result
        for k, v in result.items():
            totals[k] = totals.get(k, 0) + v
        result.update(precision_recall(result))

    totals.update(precision_recall(totals))
    return dict(config=eval_config, totals=totals)  # , details=results)


def write_wavs(reader: DBReader, convs: List[str], count_per_set: int, net_version: str, bc_sample_tracks):
    random.shuffle(convs)
    for conv in convs[0:count_per_set]:
        channel = random.choice(["A", "B"])
        convchannel = f"{conv}-{channel}"
        smoothed = reader.features.get_net_output(convchannel, "best", smooth=True)
        bcs = []
        bc_sampletrack = None
        while len(bcs) < 5:
            bc_sampletrack = random.choice(bc_sample_tracks)
            bcs = list(get_bc_samples(reader, bc_sampletrack))
        bc_audio = get_bc_audio(smoothed, reader, bcs)
        print("evaluating conv {} with bc samples from {}".format(convchannel, bc_sampletrack))
        _orig_audio = reader.features.get_adc(convchannel)
        out_dir = os.path.join("evaluate", "out", net_version)
        os.makedirs(out_dir, exist_ok=True)
        minlen = min(_orig_audio.size, bc_audio.size)
        orig_audio = normalize_audio(_orig_audio[0:minlen].reshape((minlen, 1)))
        bc_audio = normalize_audio(bc_audio[0:minlen].reshape((minlen, 1)))
        audio = np.append(orig_audio, bc_audio, axis=1)
        # audio_cut = NumFeature(np.array([], dtype="float32").reshape((0, 2)))
        for start, end in get_talking_segments(reader, convchannel, 15):
            end += 1
            start_inx = reader.features.time_to_sample_index(_orig_audio, start)
            end_inx = reader.features.time_to_sample_index(_orig_audio, end)
            soundfile.write(os.path.join(out_dir, "SPK={}{} @{:.1f}s BC={}.wav".format(conv, channel, start,
                                                                                       bc_sampletrack.replace("-",
                                                                                                              ""))),
                            audio[start_inx:end_inx], 8000)
            # audio_cut = np.append(audio_cut, audio[start_inx:end_inx])

            # soundfile.write(os.path.join(out_dir, "{}--{}.wav".format(convchannel, bc_sampletrack)), audio_cut, 8000)


def output_bc_samples(reader: DBReader, convs: List[str]):
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


def main():
    config_path = sys.argv[1]
    _, _, version, _ = config_path.split("/")

    config = load_config(config_path)

    conversations = read_conversations(config)
    # allconversations = [convid for convlist in conversations.values() for conv in convlist for convid in
    #                    [conv + "-" + channel for channel in ["A", "B"]]]

    res = []
    eval_conversations = sorted(conversations['eval'] if 'eval' in conversations else conversations['test'])
    with Parallel(n_jobs=int(os.environ["JOBS"])) as parallel:
        for eval_config in interesting_configs():
            print(eval_config)
            res.append(evaluate_convs(parallel, config_path, eval_conversations, eval_config))

    out_dir = os.path.join("evaluate", "out", version)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent='\t')


if __name__ == "__main__":
    main()
