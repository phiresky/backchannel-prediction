import sys
from .evaluate import get_best_eval_config, evaluate_conv, precision_recall
from extract.readDB import loadDBReader, read_conversations
from tqdm import tqdm
from scipy import stats
from typing import List
import numpy  as np


def confidence(arr: List[float], p=0.95):
    mean, var, std = stats.bayes_mvs(arr)
    return mean.minmax


def meanpm(arr: List[float]):
    mean = np.mean(arr)
    minm, maxm = confidence(arr, p=0.95)
    return f"{mean:.3f}±{maxm-mean:.3f}"


def significance_better_than_mmueller():
    config_path = sys.argv[1]
    config = loadDBReader(config_path).config

    conversations = read_conversations(config)

    eval_conversations = sorted(conversations['eval'])

    def filt(res):
        [l, r] = res['config']['margin_of_error']
        return res['config']['min_talk_len'] is None and r - l < 0.41

    eval_conf = get_best_eval_config(config_path, filter=filt)
    nn = []
    for conv in tqdm(eval_conversations):
        for channel in ["A", "B"]:
            convid = f"{conv}-{channel}"
            _, res = evaluate_conv(config_path, convid, eval_conf)
            nn.append(precision_recall(res)['f1_score'])

    mmueller = 0.109
    print(f"ours ({np.mean(nn)}) is better than mmueller ({mmueller}) with p={stats.ttest_1samp(nn, mmueller).pvalue}")


def significance_better_than_random():
    config_path = sys.argv[1]
    config = loadDBReader(config_path).config
    eval_conf = get_best_eval_config(config_path, margin=(0, 1))

    conversations = read_conversations(config)

    eval_conversations = sorted(conversations['eval'])

    nn = []
    rand = []
    for conv in tqdm(eval_conversations):
        for channel in ["A", "B"]:
            convid = f"{conv}-{channel}"
            _, res = evaluate_conv(config_path, convid, eval_conf)
            _, randres = evaluate_conv(config_path, convid, {**eval_conf, 'random_baseline': {}})
            nn.append(precision_recall(res)['f1_score'])
            rand.append(precision_recall(randres)['f1_score'])

    print(f"nn: f1 = {meanpm(nn)}")
    print(f"rand: f1 = {meanpm(rand)}")
    print(f"differ: p = {stats.ttest_ind(nn, rand, equal_var=False).pvalue}")


def significance_c1_vs_c2(config_path1: str, config_path2: str):
    config1 = loadDBReader(config_path1).config
    config2 = loadDBReader(config_path2).config

    eval_conf1 = get_best_eval_config(config_path1)
    eval_conf2 = get_best_eval_config(config_path2)

    conversations = read_conversations(config1)

    eval_conversations = sorted(conversations['eval'])

    r1 = []
    r2 = []
    for conv in tqdm([*eval_conversations]):  # , *sorted(conversations['validate'])
        for channel in ["A", "B"]:
            convid = f"{conv}-{channel}"
            _, res1 = evaluate_conv(config_path1, convid, {**eval_conf1, 'min_talk_len': 5})
            _, res2 = evaluate_conv(config_path2, convid, {**eval_conf2, 'min_talk_len': 5})
            r1.append(precision_recall(res1)['f1_score'])
            r2.append(precision_recall(res2)['f1_score'])

    print(f"r1: f1 = {meanpm(r1)}")
    print(f"r2: f1 = {meanpm(r2)}")
    print(f"differ: p = {stats.ttest_ind(r1, r2, equal_var=False).pvalue}")


def significance_w2v_swb_vs_swb_plus_icsi():
    config_path1 = "trainNN/out/v050-finunified-59-g49231f9-dirty:lstm-best-features-power,pitch,ffv,word2vec_dim30/config.json"
    config_path2 = "trainNN/out/v050-finunified-60-g10e2ae6-dirty:lstm-best-features-power,pitch,ffv,word2vec_dim30_4M/config.json"
    significance_c1_vs_c2(config_path1, config_path2)


def significance_acustic_vs_acoustic_plus_linguistic():
    config_path1 = "trainNN/out/v048-finunified-15-g92ee0a9-dirty:lstm-best-features-power,pitch,ffv/config.json"
    config_path2 = "trainNN/out/v050-finunified-16-g1be124b-dirty:lstm-best-features-power,pitch,ffv,word2vec_dim30-slowbatch/config.json"
    significance_c1_vs_c2(config_path1, config_path2)


if __name__ == '__main__':
    significance_acustic_vs_acoustic_plus_linguistic()
    # significance_w2v_swb_vs_swb_plus_icsi()
    # significance_better_than_mmueller()
    # significance_better_than_random()
