import lasagne
from lasagne.layers import *


def create_network(config, BATCH_SIZE):
    sequence_length = 1200 // 10
    num_labels = config['num_labels']
    input_layer = InputLayer(shape=(BATCH_SIZE, sequence_length, 2))
    input_var = input_layer.input_var
    hidden_layer_1 = LSTMLayer(input_layer, 100)
    hidden_layer_2 = LSTMLayer(hidden_layer_1, 50)
    reshape_layer = ReshapeLayer(hidden_layer_2, (-1, 50))
    almost_output_layer = DenseLayer(reshape_layer,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = ReshapeLayer(almost_output_layer, (-1, 2))
    return locals()
