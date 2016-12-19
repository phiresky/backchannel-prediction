import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, LSTMLayer


def create_network(config, BATCH_SIZE):
    input_dim = config['input_dim']
    num_labels = config['num_labels']
    input_layer = InputLayer(shape=(BATCH_SIZE, input_dim / 2, 2))
    hidden_layer_1 = LSTMLayer(input_layer, 100)
    hidden_layer_2 = LSTMLayer(hidden_layer_1, 50, only_return_final=True)
    output_layer = DenseLayer(hidden_layer_2,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax)
    return locals()
