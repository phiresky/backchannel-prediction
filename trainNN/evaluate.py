from .train import feedforward_model, load_config
from .markuslasagne.train_func import load_network_params
import theano
import lasagne.layers
import numpy


def get_network_outputter(config, weights_file: str):
    model = feedforward_model(config)

    out_layer = model['output_layer']
    load_network_params(out_layer, weights_file)
    layers = lasagne.layers.get_all_layers(out_layer)
    y = lasagne.layers.get_output(out_layer, deterministic=True)
    f = theano.function([layers[0].input_var], y)
    return f

def get_best_network_outputter(config):
    stats = config['train_output']['stats']
    best = min(stats.values(), key=lambda item: item['validation_error'])
    return get_network_outputter(config['train_config'], best['weights'])

if __name__ == "__main__":
    config = load_config("trainNN/out/latest/config.json")
    f = get_best_network_outputter(config)
    print("{}".format(f([numpy.repeat([1.], 162)])))
