assert __package__, "run as python3 -m trainNN.train"

import lasagne
from lasagne.layers import InputLayer, DenseLayer
from .markuslasagne import train_func, misc_func
import json
import sys
import numpy
from functools import partial
import theano.tensor
import os.path
import subprocess
from distutils.dir_util import mkpath

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


def feedforward_model(config):
    input_dim = config['input_dim']
    num_labels = config['num_labels']
    input_layer = InputLayer(shape=(BATCH_SIZE, input_dim))
    hidden_layer_1 = DenseLayer(input_layer,
                                num_units=100,
                                nonlinearity=lasagne.nonlinearities.sigmoid,
                                W=lasagne.init.Constant(0))
    hidden_layer_2 = DenseLayer(hidden_layer_1,
                                num_units=50,
                                nonlinearity=lasagne.nonlinearities.sigmoid,
                                W=lasagne.init.Constant(0))
    output_layer = DenseLayer(hidden_layer_2,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax)
    return locals()


def load_config(config_path: str):
    misc_func.myPrint("loading config file " + config_path)
    with open(config_path) as config_file:
        return json.load(config_file)


def train():
    version = subprocess.check_output("git describe --dirty", shell=True).decode('ascii').strip()

    out_dir = os.path.join("out", version)
    if os.path.isdir(out_dir):
        print("Output directory {} already exists, aborting".format(out_dir))
        sys.exit(1)
    mkpath(out_dir)
    LOGFILE = os.path.join(out_dir, "train.log")

    misc_func.MyLogger.logfile = open(LOGFILE, 'a')
    config_path = sys.argv[1]
    config = load_config(config_path)

    dir = os.path.dirname(config_path)
    model = feedforward_model(config)
    train_func.train_network(
        network=model['output_layer'],
        input_var=model['input_layer'].input_var,
        target_var=theano.tensor.ivector('targets'),
        scheduling_method="fuzzy_newbob",
        scheduling_params=(0.5, 0.00000001),
        update_method="adadelta",
        # learning_rate=0.01,
        iterate_minibatches_train=load_numpy_file(os.path.join(dir, config['files']['train']), config['input_dim']),
        iterate_minibatches_validate=load_numpy_file(os.path.join(dir, config['files']['validate']),
                                                     config['input_dim']),
        output_prefix=os.path.join(out_dir, "epoch")
    )


if __name__ == "__main__":
    train()
