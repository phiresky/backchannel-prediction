from train import feedforward_model, load_config
from markuslasagne.train_func import load_network_params
from markuslasagne.misc_func import myPrint
from jrtk.preprocessing import NumFeature
from functools import partial

import lasagne.layers


def get_network_output(network: lasagne.layers.DenseLayer, inp: NumFeature) -> NumFeature:
    layers = lasagne.layers.get_all_layers(network)
    return lasagne.layers.get_output(layers, inp)


def get_network_outputter(config_file: str, weights_file: str):
    config = load_config(config_file)
    model = feedforward_model(config)

    out_layer = model['output_layer']
    load_network_params(out_layer, weights_file)
    return partial(get_network_output, out_layer)


if __name__ == "__main__":
    get_network_outputter("../extract_pfiles_python/out/v08-pitchnormalization3-context40/train-config.json",
                          "out/v08-pitchnormalization3-1-g03a8cbe-dirty/epoch-003.pkl")
