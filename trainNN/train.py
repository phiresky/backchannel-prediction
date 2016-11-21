import lasagne
from lasagne.layers import InputLayer, DenseLayer
from markuslasagne import train_func, misc_func
import json
import sys
import numpy
from functools import partial
import theano.tensor
import os.path
import time

BATCH_SIZE = 256
NUM_EPOCHS = 10000
out_dir = "out"

stamp = time.strftime("%Y-%m-%d-%H:%M:%S")
LOGFILE = "{}-{}.log".format("train01", stamp)

misc_func.MyLogger.logfile = open(LOGFILE, 'a')


def iterate_minibatches(batchsize, data, input_dim):
    frames, dim = data.shape
    unused = frames % batchsize
    if unused != 0:
        misc_func.myPrint("ignoring last {} training samples".format(unused))
    assert dim == input_dim + 1
    for i in range(0, frames // batchsize):
        slice = data[i * batchsize:(i + 1) * batchsize]
        yield slice[:, :input_dim], slice[:, input_dim].astype("int32")


def load_numpy_file(fname, input_dim):
    misc_func.myPrint("loading numpy file " + fname)
    data = numpy.load(fname)['data']
    return partial(iterate_minibatches, BATCH_SIZE, data, input_dim)


config_path = sys.argv[1]
with open(config_path) as config_file:
    config = json.load(config_file)

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

dir = os.path.dirname(config_path)

train_func.train_network(
    network=output_layer,
    input_var=input_layer.input_var,
    target_var=theano.tensor.ivector('targets'),
    scheduling_method="fuzzy_newbob",
    update_method="adadelta",
    iterate_minibatches_train=load_numpy_file(os.path.join(dir, config['files']['train']), input_dim),
    iterate_minibatches_validate=load_numpy_file(os.path.join(dir, config['files']['validate']), input_dim),
    output_prefix="out"
)
