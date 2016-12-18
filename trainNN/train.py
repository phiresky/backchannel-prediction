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


def iterate_minibatches(batchsize, inputs, outputs):
    frames, dim = inputs.shape
    indices = numpy.arange(len(inputs))
    numpy.random.shuffle(indices)
    for i in range(0, frames // batchsize):
        elements = indices[i * batchsize:(i + 1) * batchsize]
        yield inputs[elements], outputs[elements]


def load_numpy_file(fname):
    misc_func.myPrint("loading numpy file " + fname)
    data = numpy.load(fname)['data']
    return data


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
    input_dim = train_config['input_dim']
    train_data = load_numpy_file(os.path.join(dir, train_config['files']['train']))
    validate_data = load_numpy_file(os.path.join(dir, train_config['files']['validate']))
    gaussian = False
    if gaussian:
        train_inputs, train_outputs = train_data[:, :input_dim], train_data[:, input_dim] / 1.33
        validate_inputs, validate_outputs = validate_data[:, :input_dim], validate_data[:, input_dim] / 1.33
    else:
        train_inputs, train_outputs = train_data[:, :input_dim], train_data[:, input_dim].astype("int32")
        validate_inputs, validate_outputs = validate_data[:, :input_dim], validate_data[:, input_dim].astype("int32")

    stats_generator = train_func.train_network(
        network=model['output_layer'],
        scheduling_method=None,
        # scheduling_params=(0.8, 0.000001),
        update_method="sgd",
        num_epochs=1000,
        learning_rate_num=1,
        iterate_minibatches_train=partial(iterate_minibatches, BATCH_SIZE, train_inputs, train_outputs),
        iterate_minibatches_validate=partial(iterate_minibatches, BATCH_SIZE, validate_inputs, validate_outputs),
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
