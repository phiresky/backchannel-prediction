import json
import sys
import random
import soundfile
import os
import numpy as np
from typing import List, Tuple, Iterator
from extract.readDB import loadDBReader, DBReader, swap_speaker, read_conversations, bc_to_category
from extract.feature import Feature, Audio
from extract.util import load_config, filter_ranges, get_talking_segments, get_monologuing_segments
from tqdm import tqdm
import functools
from itertools import product
import logging
from pprint import pformat

os.environ['JOBLIB_START_METHOD'] = 'forkserver'
from joblib import Parallel, delayed


# the word-aligned beginning of the bc is predicted
def predict_bcs(reader: DBReader, smoothed_net_output: Feature, threshold: float, at_start: bool, offset=0.0):
    for start, end in get_larger_threshold(smoothed_net_output, reader, threshold):
        if end - start < 0.02:
            continue
        if at_start:
            yield start + offset
        else:
            yield reader.get_max_time(smoothed_net_output, start, end) + offset


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


def get_larger_threshold(feat: Feature, reader: DBReader, threshold=0.5):
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
        yield feat.sample_index_to_time(start), feat.sample_index_to_time(end)
        inx = end + 1


def normalize_audio(sampletrack_audio, maxamplitude=1.0):
    max_amplitude = max(float(abs(sampletrack_audio.max())), float(abs(sampletrack_audio.min())))
    if max_amplitude == 0:
        return sampletrack_audio
    multi = (32767 * maxamplitude) / max_amplitude
    if multi < 1:
        return sampletrack_audio
    return (np.asarray(sampletrack_audio) * multi).astype("int16")


# is this nearer than that from other?
def nearer(this: float, that: float, other):
    return abs(this - other) < abs(that - other)


def bc_is_within_margin_of_error(predicted: float, correct: float, margin: Tuple[float, float]):
    return correct + margin[0] <= predicted <= correct + margin[1]


def random_predictor(reader: DBReader, convid: str, config: dict):
    shuffle_smart = config['random_baseline'].get("shuffle_in_talklen", False)
    frequency = config['random_baseline'].get("frequency", 1)
    # random.seed(convid)
    utts = list(reader.get_utterances(swap_speaker(convid)))
    mintime = float(utts[0][1]['from'])
    maxtime = float(utts[-1][1]['to'])
    bcs = reader.get_backchannels(utts)
    bcount = int(round(len(bcs) * frequency))
    if shuffle_smart:
        segs = get_monologuing_segments(reader, convid, 10)
        bcstarts = [reader.getBcRealStartTime(bc) for bc, _ in bcs]
        predicted = []
        for start, stop in segs:
            count = len(list(filter_ranges(bcstarts, [(start, stop)])))
            predicted += [random.uniform(start, stop) for _ in range(count)]
    else:
        predicted = [random.uniform(mintime, maxtime) for _ in range(bcount)]
    predicted.sort()
    return predicted


@functools.lru_cache(maxsize=476 + 400)
def cached_smoothed_netout(config_path, convid, epoch, smoother):
    return loadDBReader(config_path).features.smooth(convid, epoch, smoother)


class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))


def get_predictions(config_path: str, convid: str, eval_config: dict):
    reader = loadDBReader(config_path)
    smoothed = cached_smoothed_netout(config_path, convid, eval_config['epoch'], Hashabledict(eval_config['smoother']))
    return list(predict_bcs(reader, 1 - smoothed[:, [0]], eval_config['threshold'], eval_config['at_start']))


@functools.lru_cache()
def get_best_eval_config(config_path: str, margin: Tuple[float, float]):
    import os.path
    _, _, version, _ = config_path.split("/")
    eval_path = os.path.join("evaluate/out", version, "results.json")
    with open(eval_path) as f:
        results = json.load(f)
    results = [result for result in results if np.allclose(result['config']['margin_of_error'], margin)]
    best = max(results, key=lambda res: res['totals'].get('valid', res['totals'])['f1_score'])['config']
    print("best eval conf:")
    print(best)
    return best


def evaluate_conv(config_path: str, convid: str, config: dict):
    reader = loadDBReader(config_path)
    if reader.config['extract_config'].get('categories', None) is not None:
        return evaluate_conv_multiclass(config_path, convid, config)
    bc_convid = swap_speaker(convid)
    correct_bcs = [reader.getBcRealStartTime(utt) for utt, uttInfo in
                   reader.get_backchannels(list(reader.get_utterances(bc_convid)))]

    if 'random_baseline' in config:
        predicted_bcs = random_predictor(reader, convid, config)
    else:
        if 'sigma_ms' in config:
            if 'smoother' in config:
                raise Exception('conflicting options: smoother and sigma')
            smoother = {'type': 'gauss', 'sigma_ms': config['sigma_ms'], 'cutoff_sigma': 2}
        else:
            smoother = config['smoother']
        net_output = cached_smoothed_netout(config_path, convid, config["epoch"], Hashabledict(smoother))
        net_output = 1 - net_output[:, [0]]
        predicted_bcs = list(
            predict_bcs(reader, net_output, threshold=config['threshold'], at_start=config['at_start']))
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


def evaluate_conv_multiclass(config_path: str, convid: str, config: dict):
    # ooh boy this code be stupdid
    reader = loadDBReader(config_path)
    bc_convid = swap_speaker(convid)
    _correct_bcs = [(reader.getBcRealStartTime(utt), bc_to_category(reader, uttInfo)) for
                    utt, uttInfo in
                    reader.get_backchannels(list(reader.get_utterances(bc_convid)))]
    correct_bcs = [time for time, _ in _correct_bcs]
    correct_categories = [cat for _, cat in _correct_bcs]

    if 'sigma_ms' in config:
        if 'smoother' in config:
            raise Exception('conflicting options: smoother and sigma')
        smoother = {'type': 'gauss', 'sigma_ms': config['sigma_ms']}
    else:
        smoother = config['smoother']
    net_output = cached_smoothed_netout(config_path, convid, config["epoch"], Hashabledict(smoother))
    any_predictor = 1 - net_output[:, [0]]
    predicted_bcs = list(predict_bcs(reader, any_predictor, threshold=config['threshold'], at_start=config['at_start']))
    predicted_count = len(predicted_bcs)
    predicted_inx = 0
    predicted_categories = [np.argmax(net_output[net_output.time_to_sample_index(time)][1:]) + 1 for
                            time in
                            predicted_bcs]
    if predicted_count > 0:
        for correct_bc in correct_bcs:
            while predicted_inx < predicted_count - 1 and nearer(predicted_bcs[predicted_inx + 1],
                                                                 predicted_bcs[predicted_inx], correct_bc):
                predicted_inx += 1
            if bc_is_within_margin_of_error(predicted_bcs[predicted_inx], correct_bc, config['margin_of_error']):
                predicted_bcs[predicted_inx] = correct_bc

    correct = {time: category for time, category in zip(correct_bcs, correct_categories)}
    predicted = {time: category for time, category in zip(predicted_bcs, predicted_categories)}

    if config['min_talk_len'] is not None:
        segs = list(get_monologuing_segments(reader, convid, min_talk_len=config['min_talk_len']))
        predicted_bcs = list(filter_ranges(predicted_bcs, segs))
        correct_bcs = list(filter_ranges(correct_bcs, segs))

    category_count = len(reader.categories.keys()) + 1
    confusion = np.zeros((category_count, category_count), dtype=np.int32)

    for time in correct_bcs:
        confusion[correct[time]][predicted.get(time, 0)] += 1
    for time in predicted_bcs:
        if time not in correct:
            confusion[0][predicted[time]] += 1
    # https://www.wikiwand.com/en/Precision_and_recall
    selected = set(predicted_bcs)
    relevant = set(correct_bcs)
    true_positives = selected & relevant
    false_positives = selected - relevant
    false_negatives = relevant - selected

    return convid, dict(selected=len(selected), relevant=len(relevant), true_positives=len(true_positives),
                        false_positives=len(false_positives), false_negatives=len(false_negatives),
                        confusion_matrix=confusion)


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


def moving_margins(margin: Tuple[float, float], count=50, span=2.0):
    arr = np.array(margin)
    for offset in np.linspace(-span / 2, span / 2, count):
        yield tuple(arr + offset)


default_config = dict(margin_of_error=(-0.5, 0.5), threshold=0.6, epoch="best", min_talk_len=10,
                      smoother=dict(type='gauss-bla', sigma_ms=240, cutoff_sigma=0.7), at_start=False)


def general_interesting_configs(config):
    # http://eprints.eemcs.utwente.nl/22780/01/dekok_2012_surveyonevaluation.pdf
    interesting_margins = [(-0.1, 0.5), (-0.3, 0.3), (-0.5, 0.5), (-0.4, 0.6), (-0.6, 0.4), (0, 1), (-1, 0),
                           (-0.2, 0.2)]
    interesting_thresholds = [0.5, 0.6, 0.65, 0.7]
    interesting_talk_lens = [None, 0, 5, 10]
    for margin, threshold, talk_len in product(interesting_margins, interesting_thresholds, interesting_talk_lens):
        yield {**default_config,
               **dict(margin_of_error=margin, threshold=threshold, epoch="best", min_talk_len=talk_len)}


def smoother_interesting(config):
    smoothers = [
        *[dict(type=f"gauss-cutoff-{cutoff}σ", sigma_ms=s, cutoff_sigma=cutoff) for s in np.linspace(200, 400, 3) for
          cutoff in np.linspace(0, 3, 7)],
        *[dict(type='exponential', factor=s) for s in np.linspace(0.03, 0.06, 4)]
    ]
    margins = moving_margins((-0.25, 0.75), count=5, span=0.5)
    thresholds = np.linspace(0.5, 0.8, 5)
    for smoother, margin, threshold in product(smoothers, margins, thresholds):
        yield {**default_config,
               **dict(margin_of_error=margin, smoother={**smoother, 'type': smoother['type'] + ".s"},
                      threshold=threshold,
                      at_start=True)}
        yield {**default_config,
               **dict(margin_of_error=margin, smoother={**smoother, 'type': smoother['type'] + ".m"},
                      threshold=threshold,
                      at_start=False)}


def smoother_specific_interesting(config):
    smoothers = [
        # *[dict(type=f"gauss-cutoff-{cutoff}σ", sigma_ms=s, cutoff_sigma=cutoff) for s in np.linspace(250, 350, 3) for
        #  cutoff in np.linspace(0.5, 1.7, 7)],
        *[dict(type='exponential', factor=s) for s in np.linspace(0.03, 0.06, 4)]
    ]
    margins = moving_margins((-0.1, 0.5), 5, 0.2)
    thresholds = np.linspace(0.575, 0.725, 6)
    for smoother, margin, threshold in product(smoothers, margins, thresholds):
        # yield {**default_config,
        #       **dict(margin_of_error=margin, smoother={**smoother, 'type': smoother['type'] + ".s"},
        #              threshold=threshold,
        #              at_start=True)}
        yield {**default_config,
               **dict(margin_of_error=margin, smoother={**smoother, 'type': smoother['type'] + ".m"},
                      threshold=threshold,
                      at_start=False)}


def margin_test_configs(config):
    for margin in moving_margins((0.2, 0.6)):
        yield {**default_config, **dict(margin_of_error=margin)}

        # for margin in moving_margins((-0.5, 0.5)):
        #    yield {**default_config, **dict(margin_of_error=margin)}


def epoch_test_configs(config):
    epochs = config['train_output']['stats'].keys()
    for i, epoch in enumerate(epochs):
        if i > 100:
            continue
        yield {**default_config, **dict(epoch=epoch)}


def threshold_test_configs(config):
    for threshold in np.linspace(0.50, 0.9, 60):
        yield {**default_config, **dict(threshold=threshold)}


def min_talk_len_configs(config):
    for talk_len in [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        yield {**default_config, **dict(min_talk_len=talk_len)}


def smoothing_test_configs(config):
    for sigma_ms in range(0, 1000 + 1, 25):
        yield {**default_config, 'smoother': {**default_config['smoother'], 'sigma_ms': sigma_ms}}


def detailed_analysis(config):
    yield from min_talk_len_configs(config)
    yield from general_interesting_configs(config)
    yield from margin_test_configs(config)
    yield from threshold_test_configs(config)
    yield from smoothing_test_configs(config)
    # yield from epoch_test_configs(config)


def evaluate_convs(parallel, config_path: str, convs: List[str], eval_config: dict, showprog=False):
    import trainNN.evaluate
    totals = {}
    results = {}
    if "weights_file" not in eval_config and 'random_baseline' not in eval_config and eval_config["epoch"] == "best":
        eval_config["epoch"], eval_config["weights_file"] = trainNN.evaluate.get_best_epoch(load_config(config_path))
    convids = ["{}-{}".format(conv, channel) for conv in convs for channel in ["A", "B"]]
    tasks = [delayed(evaluate_conv)(config_path, convid, eval_config) for convid in convids]
    if showprog:
        tasks = tqdm(tasks)
    for convid, result in parallel(tasks):
        results[convid] = result
        for k, v in result.items():
            if k == 'confusion_matrix':
                totals.setdefault(k, np.zeros_like(v))
            totals[k] = totals.get(k, 0) + v
        result.update(precision_recall(result))

    totals.update(precision_recall(totals))
    return dict(config=eval_config, totals=totals)  # , details=results)


def nptolist(dictionary: dict):
    for key, val in dictionary.items():
        if isinstance(val, dict):
            nptolist(dictionary[key])
        elif isinstance(val, np.ndarray):
            dictionary[key] = dictionary[key].tolist()
    return dictionary


def general_interesting_2(config):
    for thres in [0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72]:
        for cutoff in [0.6, 0.9, 1.1, 1.3, 1.5, 2]:
            for sigma in [170, 200, 250, 300, 350]:
                for min_talk_len in [None, 0, 5, 10]:
                    for margin in [*moving_margins((-0.1, 0.9), count=3, span=0.2)
                                   # , (-0.5, 0.5), (-0.2, 0.2),(-0.3, 0.3), (-0.1, 0.5)
                                   ]:
                        for at_start in [False, True]:
                            yield {**default_config, **dict(margin_of_error=margin,
                                                            threshold=thres,
                                                            smoother=dict(type=f"gauss-cutoff-{cutoff}σ",
                                                                          sigma_ms=sigma,
                                                                          cutoff_sigma=cutoff),
                                                            at_start=at_start,
                                                            min_talk_len=min_talk_len)}


def bayesian_parameters():
    from hyperopt import hp
    return dict(
        threshold=hp.uniform('threshold', 0.6, 0.8),
        cutoff=hp.uniform('cutoff', 0, 2),
        sigma_ms=hp.uniform('sigma', 170, 350),
        margin_of_error_center=hp.uniform('margin_center', 0.3, 0.5),
        at_start=hp.choice('at_start', [False, True])
    )


results = []


def make_utility_function_hyperopt(parallel, config_path, conversations_list):
    from hyperopt import STATUS_OK
    def utility_function(conf):
        margin_of_error_center = conf['margin_of_error_center']
        threshold = conf['threshold']
        cutoff = conf['cutoff']
        sigma_ms = conf['sigma_ms']
        at_start = conf.get('at_start', False)
        eval_config = {**default_config, **dict(
            margin_of_error=(margin_of_error_center - 0.5, margin_of_error_center + 0.5),
            threshold=threshold,
            smoother=dict(type=f"gauss-cutoff-{cutoff}σ", sigma_ms=sigma_ms, cutoff_sigma=cutoff),
            at_start=at_start,
            min_talk_len=10
        )}
        result = evaluate_convs(parallel, config_path, conversations_list, eval_config)
        results.append(result)
        return {'loss': -result['totals']['f1_score'], 'status': STATUS_OK, 'result': result}

    return utility_function


def make_utility_function_gpyopt(parallel, config_path, conversations_list):
    def utility_function(conf):
        [[threshold, cutoff, sigma_ms, margin_of_error_center, min_talk_len, margin_width, at_start]] = conf
        at_start = True if at_start > 0.5 else False
        eval_config = {**default_config, **dict(
            margin_of_error=(margin_of_error_center - margin_width / 2, margin_of_error_center + margin_width / 2),
            threshold=threshold,
            smoother=dict(type=f"gauss-cutoff-{cutoff:.2f}σ.{'s' if at_start else 'm'}", sigma_ms=sigma_ms,
                          cutoff_sigma=cutoff),
            at_start=at_start,
            min_talk_len=min_talk_len if min_talk_len >= 0 else None
        )}

        result = evaluate_convs(parallel, config_path, conversations_list, eval_config)
        results.append(result)
        f1 = result['totals']['f1_score']
        print(f"it={len(results)} f1={f1} with\n{pformat(eval_config)}")
        return -f1

    return utility_function


def bayesian_optimize_hyperopt(parallel, config_path, conversations_list):
    from hyperopt import fmin, tpe, Trials
    global results
    results = []
    utility = make_utility_function_hyperopt(parallel, config_path, conversations_list)
    trials = Trials()
    opt = fmin(utility, space=bayesian_parameters(), algo=tpe.suggest, max_evals=50, trials=trials)
    print(opt)
    # bo = BayesianOptimization(utility, bayesian_parameters())
    # bo.maximize(init_points=10, n_iter=100)


def gpyopt_parameters_best():
    return [
        dict(name='threshold', type='continuous', domain=(0.6, 0.8)),
        dict(name='cutoff', type='continuous', domain=(0, 2)),
        dict(name='sigma_ms', type='continuous', domain=(200, 350)),
        dict(name='margin_of_error_center', type='continuous', domain=(0.4, 0.5)),
        dict(name='min_talk_len', type='continuous', domain=(5.0, 5.0)),
        dict(name='margin_width', type='continuous', domain=(1, 1)),
        dict(name='at_start', type='discrete', domain=(0, 0)),
    ]


def gpyopt_parameters_center0():
    params = gpyopt_parameters_best()
    params[3]['domain'] = (-0.4, 0.0)
    return params


# comparison with margin (-0.2, +0.2)
def gpyopt_parameters_w4():
    return [
        dict(name='threshold', type='continuous', domain=(0.6, 0.9)),
        dict(name='cutoff', type='continuous', domain=(0, 2)),
        dict(name='sigma_ms', type='continuous', domain=(150, 350)),
        dict(name='margin_of_error_center', type='continuous', domain=(0.0, 0.0)),
        dict(name='min_talk_len', type='continuous', domain=(5.0, 5.0)),
        dict(name='margin_width', type='continuous', domain=(0.4, 0.4)),
        dict(name='at_start', type='discrete', domain=(0, 1)),
    ]


# comparison with margin (-0.1, +0.5)
def gpyopt_parameters_w6():
    return [
        dict(name='threshold', type='continuous', domain=(0.6, 0.8)),
        dict(name='cutoff', type='continuous', domain=(0, 2)),
        dict(name='sigma_ms', type='continuous', domain=(150, 350)),
        dict(name='margin_of_error_center', type='continuous', domain=(0.2, 0.2)),
        dict(name='min_talk_len', type='continuous', domain=(5.0, 5.0)),
        dict(name='margin_width', type='continuous', domain=(0.6, 0.6)),
        dict(name='at_start', type='discrete', domain=(0, 1)),

    ]


def gpyopt_parameters_mmueller():
    return [
        dict(name='threshold', type='continuous', domain=(0.6, 0.9)),
        dict(name='cutoff', type='continuous', domain=(0, 2)),
        dict(name='sigma_ms', type='continuous', domain=(150, 350)),
        dict(name='margin_of_error_center', type='continuous', domain=(-0.5, 0.0)),  # domain=(-2,2) for offline),
        dict(name='min_talk_len', type='continuous', domain=(-1.0, -1.0)),
        dict(name='margin_width', type='continuous', domain=(0.4, 0.4)),
        dict(name='at_start', type='discrete', domain=(0, 1)),
    ]


def gpyopt(parallel, config_path, conversations_list, params):
    from GPyOpt.methods import BayesianOptimization
    global results
    results = []
    utility = make_utility_function_gpyopt(parallel, config_path, conversations_list)
    bo = BayesianOptimization(utility, params, verbosity=True, initial_design_numdata=len(params) * 2)
    bo.run_optimization(max_iter=200, verbosity=True)
    print(f"done. best: {bo.fx_opt}")
    return results


def gpyopt_all(parallel, config_path, convos_valid, convos_eval):
    for params in [
        gpyopt_parameters_center0,
        gpyopt_parameters_mmueller,
        gpyopt_parameters_best,
        gpyopt_parameters_w4,
        gpyopt_parameters_w6
    ]:
        print(f"searching in {params.__name__}")
        results = gpyopt(parallel, config_path, convos_valid, params())
        best = max(results, key=lambda res: res['totals']['f1_score'])
        best_eval = evaluate_convs(parallel, config_path, convos_eval, best['config'])
        best.update(dict(config=best['config'], totals={'eval': best_eval['totals'], 'valid': best['totals']}))
        yield [nptolist(res) for res in results]


def stat(config_path: str):
    reader = loadDBReader(config_path)
    convs = read_conversations(reader.config)
    convids = ["{}-{}".format(conv, channel) for conv in convs['eval'] for channel in ["A", "B"]]
    total_s = 0
    mono_s = 0
    bc_count = 0
    utt_count = 0
    mono_bc_count = 0
    for convid in convids:
        utts = list(reader.get_utterances(convid))
        utt_count += len(utts)
        start = float(utts[0][1]['from'])
        end = float(utts[-1][1]['to'])
        total_s += end - start
        monosegs = list(get_monologuing_segments(reader, convid, min_talk_len=5))
        mono_s += sum(end - start for (start, end) in monosegs)
        bcs = list(reader.get_backchannels(utts))
        bcs = [reader.getBcRealStartTime(utt) for utt, uttInfo in bcs]
        bc_count += len(bcs)
        bcs = list(filter_ranges(bcs, monosegs))
        mono_bc_count += len(bcs)
    print(dict(total_s=total_s, mono_s=mono_s, bc_count=bc_count, utt_count=utt_count, mono_bc_count=mono_bc_count))


do_detailed_analysis = False
manual_analysis = False


def main():
    config_path = sys.argv[1]
    # return stat(config_path)
    _, _, version, _ = config_path.split("/")
    out_dir = os.path.join("evaluate", "out", version)
    if os.path.isdir(out_dir):
        print("Output directory {} already exists, aborting".format(out_dir))
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    logging.root.handlers.clear()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[
                            # logging.FileHandler(LOGFILE),
                            logging.StreamHandler()
                        ])
    config = load_config(config_path)

    conversations = read_conversations(config)

    res = []
    eval_conversations = sorted(conversations['eval'])
    valid_conversations = sorted(conversations['validate'])
    do_baseline = config['eval_config'].get('do_random_baseline', None)
    if do_detailed_analysis:
        confs = list(detailed_analysis(config))
    else:
        confs = list(general_interesting_2(config))
    with Parallel(n_jobs=int(os.environ.get('JOBS', '1'))) as parallel:
        print(f"filling caches...")
        eval_config = default_config
        if do_baseline is not None:
            eval_config['random_baseline'] = do_baseline
        evaluate_convs(parallel, config_path, [*valid_conversations, *eval_conversations],
                       default_config, showprog=True)
        if not manual_analysis:
            itera = gpyopt_all(parallel, config_path, valid_conversations, eval_conversations)
            for inx, r in enumerate(itera):
                print(f" itera {inx} ({len(r)} results)")
                print(f"bayesian search done, checking scores on eval data set")
                res.extend(r)
                with open(os.path.join(out_dir, "results.json"), "w") as f:
                    json.dump(res, f, indent='\t')
            return
        else:
            # manual search
            for inx, eval_config in enumerate(confs):
                if do_baseline is not None:
                    eval_config['random_baseline'] = do_baseline
                print(f"\n{inx}/{len(confs)}: {eval_config}\n")
                ev = evaluate_convs(parallel, config_path, eval_conversations, eval_config)
                va = evaluate_convs(parallel, config_path, valid_conversations, eval_config)
                res.append(nptolist(dict(config=ev['config'], totals={'eval': ev['totals'], 'valid': va['totals']})))

                with open(os.path.join(out_dir, "results.json"), "w") as f:
                    json.dump(res, f, indent='\t')


if __name__ == "__main__":
    main()
