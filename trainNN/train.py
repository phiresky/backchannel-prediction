from .markuslasagne import train_func, misc_func
import json
import sys
import numpy
from functools import partial
import theano.tensor
import os.path
import subprocess
from distutils.dir_util import mkpath
import contextlib
from .network_model import create_network
import inspect
import shutil
from extract_pfiles_python.util import load_config
import itertools
from typing import List, Tuple

BATCH_SIZE = 256
NUM_EPOCHS = 10000


def iterate_minibatches(batchsize, groups: List[Tuple[int, int]], inputs, outputs):
    sequence_length = groups[0].stop - groups[0].start
    group_count = len(groups)
    numpy.random.shuffle(groups)
    for i in range(0, group_count // batchsize):
        ranges = groups[i * batchsize:(i + 1) * batchsize]
        yield [inputs[range] for range in ranges], numpy.array([outputs[range] for range in ranges]).reshape(
            (batchsize * sequence_length,))


def load_numpy_file(fname):
    misc_func.myPrint("loading numpy file " + fname)
    data = numpy.load(fname)['data']
    return data


def train():
    version = subprocess.check_output("git describe --dirty", shell=True).decode('ascii').strip()

    out_dir = os.path.join("trainNN", "out", version)
    if os.path.isdir(out_dir):
        print("Output directory {} already exists, aborting".format(out_dir))
        sys.exit(1)
    mkpath(out_dir)
    LOGFILE = os.path.join(out_dir, "train.log")

    model_file = os.path.join(out_dir, "network_model.py")
    shutil.copyfile(inspect.getsourcefile(create_network), model_file)

    misc_func.MyLogger.logfile = open(LOGFILE, 'a')
    misc_func.myPrint("version={}".format(version))
    config_path = sys.argv[1]
    config = load_config(config_path)
    train_config = config['train_config']

    dir = os.path.dirname(config_path)
    model = create_network(train_config, BATCH_SIZE)
    input_dim = train_config['input_dim']
    gaussian = False
    if gaussian:
        train_data = load_numpy_file(os.path.join(dir, train_config['files']['train']))
        validate_data = load_numpy_file(os.path.join(dir, train_config['files']['validate']))
        train_inputs, train_outputs = train_data[:, :input_dim], train_data[:, input_dim]
        validate_inputs, validate_outputs = validate_data[:, :input_dim], validate_data[:, input_dim]
    else:
        groups = {}
        inputs = {}
        outputs = {}
        batchers = {}
        for t in 'train', 'validate':
            with open(os.path.join(dir, train_config['files'][t]['ids'])) as f:
                meta = json.load(f)
            groups[t] = [slice(begin, end) for begin, end in meta['ranges']]
            inputs[t] = load_numpy_file(os.path.join(dir, train_config['files'][t]['input']))
            outputs[t] = load_numpy_file(os.path.join(dir, train_config['files'][t]['output']))
            batchers[t] = partial(iterate_minibatches, BATCH_SIZE, groups[t], inputs[t], outputs[t])

    stats_generator = train_func.train_network(
        network=model['output_layer'],
        scheduling_method=None,
        # scheduling_params=(0.8, 0.000001),
        update_method="sgd",
        num_epochs=1000,
        learning_rate_num=1,
        iterate_minibatches_train=batchers['train'],
        iterate_minibatches_validate=batchers['validate'],
        categorical_output=not gaussian,
        output_prefix=os.path.join(out_dir, "epoch")
    )
    config_out = os.path.join(out_dir, "config.json")
    for stats in stats_generator:
        for k, v in stats.items():
            v['weights'] = os.path.basename(v['weights'])
        with open(config_out, "w") as f:
            json.dump({**config, 'train_output': {
                'stats': stats,
                'source': config_path,
                'model': os.path.basename(model_file)
            }}, f, indent='\t')
        misc_func.myPrint("Wrote output to " + config_out)
    latest_path = os.path.join("trainNN", "out", "latest")
    with contextlib.suppress(FileNotFoundError):
        os.remove(latest_path)
    os.symlink(version, latest_path)


if __name__ == "__main__":
    train()
