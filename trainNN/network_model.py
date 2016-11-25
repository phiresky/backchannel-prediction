import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer


def create_network(config, BATCH_SIZE):
    input_dim = config['input_dim']
    num_labels = config['num_labels']
    input_layer = InputLayer(shape=(BATCH_SIZE, input_dim))
    drop_1 = DropoutLayer(input_layer, p=0.2)
    hidden_layer_1 = DenseLayer(drop_1,
                                num_units=100,
                                nonlinearity=lasagne.nonlinearities.sigmoid,
                                # W=lasagne.init.Constant(0)
                                )
    drop_2 = DropoutLayer(hidden_layer_1, p=0.3)
    hidden_layer_2 = DenseLayer(drop_2,
                                num_units=50,
                                nonlinearity=lasagne.nonlinearities.sigmoid,
                                # W=lasagne.init.Constant(0)
                                )
    drop_3 = DropoutLayer(hidden_layer_2, p=0.2)
    output_layer = DenseLayer(drop_3,
                              num_units=num_labels,
                              nonlinearity=lasagne.nonlinearities.softmax
                              )
    return locals()
