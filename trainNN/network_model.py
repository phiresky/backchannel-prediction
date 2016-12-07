import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer


def create_network(config, BATCH_SIZE):
    input_dim = config['input_dim']
    num_labels = config['num_labels']
    input_layer = InputLayer(shape=(BATCH_SIZE, input_dim))
    hidden_layer_1 = DenseLayer(input_layer,
                                num_units=100,
                                nonlinearity=lasagne.nonlinearities.sigmoid,
                                # W=lasagne.init.Constant(0)
                                )
    hidden_layer_2 = DenseLayer(hidden_layer_1,
                                num_units=50,
                                nonlinearity=lasagne.nonlinearities.sigmoid,
                                # W=lasagne.init.Constant(0)
                                )
    output_layer = DenseLayer(hidden_layer_2,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax
                              )
    return locals()
