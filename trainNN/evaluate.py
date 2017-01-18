import numpy
import os.path
import functools


def load_module_from_path(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("unknown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_best_epoch(config: dict):
    stats = config['train_output']['stats']
    key = min(stats.keys(), key=lambda k: stats[k]['validation_error'])
    return key, stats[key]['weights']


@functools.lru_cache(maxsize=1)
def get_network_outputter(config_path: str, key: str):
    # load modules lazily to avoid startup delay when not needed
    import theano
    import lasagne.layers
    from .train import load_config
    from .markuslasagne.train_func import load_network_params
    config = load_config(config_path)
    model_file = os.path.join(os.path.dirname(config_path), config['train_output']['model'])
    stats = config['train_output']['stats']
    if key == "best":
        key, _ = get_best_epoch(config)
    weights_file = os.path.join(os.path.dirname(config_path), stats[key]['weights'])
    model = load_module_from_path(model_file).create_network(config['train_config'], None)
    out_layer = model['output_layer']
    load_network_params(out_layer, weights_file)
    layers = lasagne.layers.get_all_layers(out_layer)
    y = lasagne.layers.get_output(out_layer, deterministic=True)
    f = theano.function([layers[0].input_var], y)
    return layers, f
