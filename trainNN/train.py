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

BATCH_SIZE = 256
NUM_EPOCHS = 10000


def iterate_minibatches(batchsize, data, input_dim):
    frames, dim = data.shape
    assert dim == input_dim + 1
    indices = numpy.arange(len(data))
    numpy.random.shuffle(indices)
    for i in range(0, frames // batchsize):
        elements = indices[i * batchsize:(i + 1) * batchsize]
        slice = data[elements]
        yield slice[:, :input_dim], slice[:, input_dim].astype("int32")


def load_numpy_file(fname, input_dim):
    misc_func.myPrint("loading numpy file " + fname)
    data = numpy.load(fname)['data']
    return partial(iterate_minibatches, BATCH_SIZE, data, input_dim)


def load_config(config_path: str):
    misc_func.myPrint("loading config file " + config_path)
    with open(config_path) as config_file:
        return json.load(config_file)


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
    stats = train_func.train_network(
        network=model['output_layer'],
        input_var=model['input_layer'].input_var,
        target_var=theano.tensor.ivector('targets'),
        scheduling_method="fuzzy_newbob",
        scheduling_params=(0.5, 0.00000001),
        update_method="adadelta",
        # learning_rate=0.01,
        iterate_minibatches_train=load_numpy_file(os.path.join(dir, train_config['files']['train']),
                                                  train_config['input_dim']),
        iterate_minibatches_validate=load_numpy_file(os.path.join(dir, train_config['files']['validate']),
                                                     train_config['input_dim']),
        output_prefix=os.path.join(out_dir, "epoch")
    )
    config_out = os.path.join(out_dir, "config.json")
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
