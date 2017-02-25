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

if __name__ == '__main__':
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
