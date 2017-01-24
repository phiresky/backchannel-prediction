import lasagne
from lasagne.layers import *


def lstm_simple(train_config):
    sequence_length = train_config['context_frames']
    input_dim = train_config['input_dim']

    num_labels = train_config['num_labels']
    batch_size = train_config.get('batch_size', None)
    layer_sizes = train_config['layer_sizes']
    out_all = {'all': True, 'single': False}[train_config['output_type']]

    input_layer = InputLayer(shape=(batch_size, sequence_length, input_dim))
    input_var = input_layer.input_var

    cur_input = input_layer
    [*hidden_layers, last_hidden_layer] = layer_sizes
    for size in hidden_layers:
        cur_input = LSTMLayer(incoming=cur_input, num_units=size)

    if out_all:
        cur_input = LSTMLayer(incoming=cur_input, num_units=last_hidden_layer)
        cur_input = ReshapeLayer(cur_input, (batch_size * sequence_length if batch_size else -1, last_hidden_layer))
    else:
        cur_input = LSTMLayer(incoming=cur_input, num_units=last_hidden_layer, only_return_final=True)

    output_layer = DenseLayer(cur_input,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax)
    # output_layer = ReshapeLayer(almost_output_layer, (batch_size * sequence_length if batch_size else -1, 2))
    return locals()


def feedforward_simple(train_config):
    sequence_length = train_config['context_frames']
    input_dim = train_config['input_dim']

    num_labels = train_config['num_labels']
    batch_size = train_config.get('batch_size', None)
    layer_sizes = train_config['layer_sizes']

    input_layer = InputLayer(shape=(batch_size, sequence_length, input_dim))
    input_var = input_layer.input_var

    input_layer_reshaped = ReshapeLayer(incoming=input_layer,
                                        shape=(-1 if batch_size is None else batch_size, sequence_length * input_dim))
    cur_input = input_layer_reshaped
    for size in layer_sizes:
        cur_input = DenseLayer(incoming=cur_input, num_units=size)

    output_layer = DenseLayer(cur_input,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax)
    return locals()
