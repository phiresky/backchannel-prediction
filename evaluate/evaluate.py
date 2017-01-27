import json
import sys
import random
import soundfile
import os
import numpy as np
from jrtk.preprocessing import NumFeature
from typing import List, Tuple, Iterator
from extract_pfiles_python.readDB import loadDBReader, DBReader, swap_speaker, read_conversations
from extract_pfiles_python.util import load_config, filter_ranges, get_talking_segments, get_monologuing_segments
from tqdm import tqdm
import functools
import trainNN.evaluate
from trainNN import train
from itertools import product

os.environ['JOBLIB_START_METHOD'] = 'forkserver'
from joblib import Parallel, delayed


def get_bc_samples(reader: DBReader, bc_filter, sampletrack="sw4687-B"):
    sampletrack_audio = reader.features.get_adc(sampletrack)
    bcs = reader.get_backchannels(list(reader.get_utterances(sampletrack)))
    for bc_id, bc_info in bcs:
        if bc_filter and not bc_filter(bc_info):
            continue
        bc_start_time = float(bc_info['from'])
        bc_audio = sampletrack_audio[f"{bc_start_time}s":f"{bc_info['to']}s"]
        bc_real_start_time = reader.getBcRealStartTime(bc_id)
        bc_start_offset = bc_real_start_time - bc_start_time
        yield bc_start_offset, bc_audio


# the word-aligned beginning of the bc is predicted
def predict_bcs(reader: DBReader, smoothed_net_output: NumFeature, threshold: float):
    for start, end in get_larger_threshold(smoothed_net_output, reader, threshold):
        peak_prediction_s = reader.get_max_time(smoothed_net_output, start, end)  # - np.average(reader.method['bc'])
        peak_prediction_s += reader.config['eval_config']['prediction_offset']
        yield peak_prediction_s


def get_bc_audio(smoothed_net_output: NumFeature, reader: DBReader,
                 bcs: List[Tuple[int, NumFeature]]):
    total_length_s = reader.features.sample_index_to_time(smoothed_net_output, smoothed_net_output.shape[0])
    total_length_audio_index = reader.features.time_to_sample_index(bcs[0][1], total_length_s)
    return get_bc_audio2(reader, total_length_audio_index, bcs, predict_bcs(reader, smoothed_net_output, threshold=0.6))


def get_bc_audio2(reader: DBReader, total_length_audio_index: float, bcs: List[Tuple[int, NumFeature]],
                  predictions: Iterator[float]):
    output_audio = NumFeature(np.zeros(total_length_audio_index, dtype='int16'),
                              samplingRate=bcs[0][1].samplingRate)

    for peak_s in predictions:
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
    search_arr = bool_arr[start_inx:]
    if len(search_arr) == 0:
        return None
    inx = np.argmax(search_arr) + start_inx
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


def normalize_audio(sampletrack_audio, maxamplitude=1.0):
    max_amplitude = max(float(abs(sampletrack_audio.max())), float(abs(sampletrack_audio.min())))
    if max_amplitude == 0:
        return sampletrack_audio
    multi = (32767 * maxamplitude) / max_amplitude
    if multi < 1:
        return sampletrack_audio
    return (sampletrack_audio * multi).astype("int16")


# is this nearer than that from other?
def nearer(this: float, that: float, other):
    return abs(this - other) < abs(that - other)


def bc_is_within_margin_of_error(predicted: float, correct: float, margin: Tuple[float, float]):
    return correct + margin[0] <= predicted <= correct + margin[1]


def evaluate_conv(config_path: str, convid: str, config: dict):
    reader = loadDBReader(config_path)
    bc_convid = swap_speaker(convid)
    correct_bcs = [reader.getBcRealStartTime(utt) for utt, uttInfo in
                   reader.get_backchannels(list(reader.get_utterances(bc_convid)))]

    net_output = reader.features.get_multidim_net_output(convid, config["epoch"], smooth=True)
    net_output = 1 - net_output[:, [0]]
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
        segs = list(get_monologuing_segments(reader, convid, min_talk_len=config['min_talk_len']))
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
        recall = 0 if stats['false_negatives'] > 0 else 1
        precision = 0 if stats['false_positives'] > 0 else 1
    else:
        precision = stats['true_positives'] / stats['selected']
        recall = stats['true_positives'] / stats['relevant']

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return dict(precision=precision, recall=recall, f1_score=f1_score)


def moving_margins(margin: Tuple[float, float], count=50, span=2):
    arr = np.array(margin)
    for offset in np.linspace(-span / 2, span / 2, count):
        yield tuple(arr + offset)


default_config = dict(margin_of_error=(-0.5, 0.5), threshold=0.6, epoch="best", min_talk_len=10)


def general_interesting_configs(config):
    # http://eprints.eemcs.utwente.nl/22780/01/dekok_2012_surveyonevaluation.pdf
    interesting_margins = [(-0.1, 0.5), (-0.5, 0.5), (0, 1), (-1, 0), (-0.2, 0.2)]
    interesting_thresholds = [0.5, 0.6, 0.65, 0.7]
    interesting_talk_lens = [None, 5, 10]
    for margin, threshold, talk_len in product(interesting_margins, interesting_thresholds, interesting_talk_lens):
        yield dict(margin_of_error=margin, threshold=threshold, epoch="best", min_talk_len=talk_len)


def margin_test_configs(config):
    for margin in moving_margins((-0.2, 0.2)):
        yield {**default_config, **dict(margin_of_error=margin)}


def epoch_test_configs(config):
    epochs = config['train_output']['stats'].keys()
    for i, epoch in enumerate(epochs):
        if i > 100:
            continue
        yield {**default_config, **dict(epoch=epoch)}


def threshold_test_configs(config):
    for threshold in np.linspace(0.55, 0.65, 30):
        yield {**default_config, **dict(threshold=threshold)}


def min_talk_len_configs(config):
    for talk_len in [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        yield {**default_config, **dict(min_talk_len=talk_len)}


def detailed_analysis(config):
    yield from min_talk_len_configs(config)
    yield from epoch_test_configs(config)
    yield from general_interesting_configs(config)
    yield from margin_test_configs(config)
    yield from threshold_test_configs(config)


def evaluate_convs(parallel, config_path: str, convs: List[str], eval_config: dict):
    totals = {}
    results = {}
    if "weights_file" not in eval_config and eval_config["epoch"] == "best":
        eval_config["epoch"], eval_config["weights_file"] = trainNN.evaluate.get_best_epoch(load_config(config_path))
    convids = ["{}-{}".format(conv, channel) for conv in convs for channel in ["A", "B"]]
    for convid, result in parallel(
            tqdm([delayed(evaluate_conv)(config_path, convid, eval_config) for convid in convids])):
        results[convid] = result
        for k, v in result.items():
            totals[k] = totals.get(k, 0) + v
        result.update(precision_recall(result))

    totals.update(precision_recall(totals))
    return dict(config=eval_config, totals=totals)  # , details=results)


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
    out_dir = os.path.join("evaluate", "out", version)
    if os.path.isdir(out_dir):
        print("Output directory {} already exists, aborting".format(out_dir))
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    config = load_config(config_path)

    conversations = read_conversations(config)

    res = []
    eval_conversations = sorted(conversations['eval'])
    valid_conversations = sorted(conversations['validate'])
    with Parallel(n_jobs=int(os.environ["JOBS"]), batch_size=1) as parallel:
        confs = list(general_interesting_configs(config))
        for inx, eval_config in enumerate(confs):
            print(f"\n{inx}/{len(confs)}: {eval_config}\n")
            ev = evaluate_convs(parallel, config_path, eval_conversations, eval_config)
            va = evaluate_convs(parallel, config_path, valid_conversations, eval_config)
            res.append(dict(config=ev['config'], totals={'eval': ev['totals'], 'valid': va['totals']}))

            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(res, f, indent='\t')


if __name__ == "__main__":
    main()
