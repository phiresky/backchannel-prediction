from .train import feedforward_model, load_config
from .markuslasagne.train_func import load_network_params
import theano
import lasagne.layers
import numpy

def get_network_outputter(config_file: str, weights_file: str):
    config = load_config(config_file)
    model = feedforward_model(config)

    out_layer = model['output_layer']
    load_network_params(out_layer, weights_file)
    layers = lasagne.layers.get_all_layers(out_layer)
    y = lasagne.layers.get_output(out_layer, deterministic=True)
    f = theano.function([layers[0].input_var], y)
    return f


if __name__ == "__main__":
    f = get_network_outputter("../extract_pfiles_python/out/v08-pitchnormalization3-context40/train-config.json",
                          "out/v08-pitchnormalization3-1-g03a8cbe-dirty/epoch-003.pkl")
    print("{}".format(f([numpy.repeat([1.], 162)])))
