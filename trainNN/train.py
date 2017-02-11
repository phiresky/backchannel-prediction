import contextlib
import inspect
import json
import os
import os.path
import shutil
import subprocess
import sys
import logging
from functools import partial
from typing import List, Tuple
import itertools
import numpy
import numpy as np
from typing import TypeVar
from extract.util import load_config, batch_list, windowed_indices
from trainNN import train_func
from . import network_model, evaluate
from extract import readDB
import functools
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import random

NUM_EPOCHS = 10000

reader = None
backchannels = None
config_path = None


def extract(utterance: Tuple[str, int, bool]):
    if len(utterance) == 2:
        utt_id, is_bc = utterance
    else:
        utt_id, copy_id, is_bc = utterance
    return readDB.extract(config_path)[utt_id, is_bc]


def extract_batch(batch: List[Tuple[Tuple[str, int, bool], List[int]]], context_frames: int,
                  input_dim: int,
                  weighted_update: bool,
                  output_all: bool):
    inputs = np.zeros((len(batch), context_frames, input_dim), dtype='float32')
    outputs = np.zeros((len(batch),), dtype='int32')  # if not output_all else np.empty((len(batch), context_frames),
    #                                 dtype='int32')
    weights = np.ones((len(batch),), dtype='float32')
    for index, (utterance_id, indices) in enumerate(batch):
        cur_inputs, cur_outputs = extract(utterance_id)
        inputs[index] = cur_inputs[indices]
        outputs[index] = cur_outputs[indices][0]  # if not output_all else cur_outputs[indices][:, 0]
        if weighted_update:
            weights[index] = utterance_id[1]
    return inputs, outputs, weights


def iterate_minibatches(train_config, all_elements, output_all, random=True):
    batchsize = train_config['batch_size']
    context_frames = train_config['context_frames']
    input_dim = train_config['input_dim']
    weighted_update = train_config.get('balance_method', None) == "weighted"
    if random:
        numpy.random.shuffle(all_elements)
    for _, batch in batch_list(all_elements, batchsize, include_last_partial=False):
        yield extract_batch(batch, context_frames, input_dim, weighted_update=weighted_update, output_all=output_all)


# randomize every batch only once at start. significantly improves performance but might make training worse
def iterate_faster_minibatches(train_config, all_elements, output_all):
    logging.debug("shuffling batches")
    batches = list(tqdm(iterate_minibatches(train_config, all_elements, output_all)))
    logging.debug("shuffling done")

    def iter():
        random.shuffle(batches)
        return batches

    return iter


def load_numpy_file(fname):
    logging.info("loading numpy file " + fname)
    data = numpy.load(fname)['data']
    return data


def benchmark_batcher(batcher):
    print("benchmarking batcher")
    before = time.perf_counter()
    for batch in tqdm(batcher()):
        pass
    after = time.perf_counter()
    print(f"batching took {after-before:.2f}s")


def train():
    global reader
    global backchannels
    global config_path

    config_path = sys.argv[1]
    config = load_config(config_path)
    version = subprocess.check_output("git describe --dirty", shell=True).decode('ascii').strip()

    if config_path.startswith("trainNN/out"):
        out_dir = os.path.dirname(config_path)
        print("Continuing training from folder " + out_dir)
        load_stats = config['train_output']['stats']
        load_epoch = max([int(epoch) for epoch in load_stats.keys()])
        load_params = os.path.join(out_dir, config['train_output']['stats'][str(load_epoch)]['weights'])
        print(f"Continuing training from folder {out_dir}, epoch={load_epoch}, params={load_params}")
        config.setdefault('train_output_old', {})[load_epoch] = config['train_output']
    else:
        load_epoch = -1
        load_stats = {}
        load_params = None
        out_dir = os.path.join("trainNN", "out", version + ":" + config['name'])
        if os.path.isdir(out_dir):
            print("Output directory {} already exists, aborting".format(out_dir))
            sys.exit(1)
        os.makedirs(out_dir, exist_ok=True)
    LOGFILE = os.path.join(out_dir, "train.log")
    logging.root.handlers.clear()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[
                            logging.FileHandler(LOGFILE),
                            logging.StreamHandler()
                        ])

    logging.debug("version={}:{}".format(version, config['name']))
    reader = readDB.loadDBReader(config_path)
    train_config = config['train_config']
    context_stride = train_config['context_stride']
    context_frames = int(train_config['context_ms'] / 10 / context_stride)
    train_config['context_frames'] = context_frames

    gaussian = train_config['gaussian']
    out_all = {'all': True, 'single': False}[train_config['output_type']]
    if gaussian:
        raise Exception("not implemented")
        # train_data = load_numpy_file(os.path.join(dir, train_config['files']['train']))
        # validate_data = load_numpy_file(os.path.join(dir, train_config['files']['validate']))
        # train_inputs, train_outputs = train_data[:, :input_dim], train_data[:, input_dim]
        # validate_inputs, validate_outputs = validate_data[:, :input_dim], validate_data[:, input_dim]
    else:
        batchers = {}
        for t in 'train', 'validate':
            # with open(os.path.join(dir, train_config['files'][t]['ids'])) as f:
            #    meta = json.load(f)
            # groups = [slice(begin, end) for begin, end in meta['ranges']]
            # inputs = load_numpy_file(os.path.join(dir, train_config['files'][t]['input']))
            # outputs = load_numpy_file(os.path.join(dir, train_config['files'][t]['output']))
            convos = readDB.read_conversations(config)
            balance_method = train_config.get('balance_method', None)
            uttids = [bc for bc in readDB.all_uttids(config_path, convos[t]) if extract(bc) is not None]
            if balance_method is None:
                backchannels = list(readDB.balance_data(config_path, uttids))
            elif balance_method == "weighted":
                backchannels = list(readDB.get_balanced_weights(config_path, uttids))
            else:
                raise Exception(f"unknown balance method {balance_method}")
            input_dim = extract(backchannels[0])[0].shape[1]
            logging.debug(f"set input dim to {input_dim}")
            inxtoname = {**{v: k for k, v in reader.category_to_index.items()}, 0: None}

            train_config['input_dim'] = input_dim
            if config['extract_config'].get('categories', None) is not None:
                category_names = [inxtoname[inx] for inx in range(len(reader.categories) + 1)]
                train_config['category_names'] = category_names
                train_config['num_labels'] = len(category_names)
            else:
                train_config['num_labels'] = 2
            logging.debug(f"input dim = {input_dim}")
            context_stride = train_config['context_stride']
            context_length = int(train_config['context_ms'] / 10 / context_stride)
            sequence_length = int((reader.method['nbc'][1] - reader.method['nbc'][0]) * 1000 / 10)
            inner_indices = windowed_indices(sequence_length, context_length, context_stride)
            all_elements = list(itertools.product(backchannels, inner_indices))
            batchers[t] = iterate_faster_minibatches(train_config, all_elements, out_all)
            before = time.perf_counter()
            before_cpu = time.process_time()
            logging.debug("loading data into ram")
            i = 0
            for backchannel in tqdm(backchannels):
                extract(backchannel)
            logging.debug(
                f"loading data took {time.perf_counter () - before:.3f}s (cpu: {time.process_time()-before_cpu:.3f}s)")

    create_network = getattr(network_model, train_config['model_function'])
    model = create_network(train_config)
    out_layer = model['output_layer']

    resume_parameters = train_config.get('resume_parameters', None)
    finetune_config = train_config.get("finetune", None)
    if finetune_config is not None:
        import lasagne
        if load_params is not None or resume_parameters is not None:
            raise Exception("cant finetune and load")
        ft_config_path = finetune_config['config']
        epoch = finetune_config['epoch']
        which_layers = finetune_config['layers']
        ft_layers, _ = evaluate.get_network_outputter(ft_config_path, epoch, batch_size=250)
        layers = lasagne.layers.get_all_layers(out_layer)
        for inx, (layer_config, layer, ft_layer) in enumerate(zip(which_layers, layers, ft_layers)):
            do_load = layer_config['load']
            do_freeze = layer_config['freeze']
            if do_load:
                for param, ft_param in zip(layer.get_params(), ft_layer.get_params()):
                    param.set_value(ft_param.get_value())
                logging.info(f"loaded layer {inx} ({ {repr(p): p.get_value().shape for p in layer.get_params()} })")
            if do_freeze:
                logging.info(f"freezing layer {inx}")
                train_func.freeze(layer)

    stats_generator = train_func.train_network(
        network=out_layer,
        twodimensional_output=False,
        scheduling_method=None,
        start_epoch=load_epoch + 1,
        resume=load_params if load_params is not None else resume_parameters,
        l2_regularization=train_config.get("l2_regularization", None),
        # scheduling_params=(0.8, 0.000001),
        update_method=train_config['update_method'],
        num_epochs=train_config['epochs'],
        learning_rate_num=train_config['learning_rate'],
        iterate_minibatches_train=batchers['train'],
        iterate_minibatches_validate=batchers['validate'],
        categorical_output=not gaussian,
        output_prefix=os.path.join(out_dir, "epoch")
    )
    config_out = os.path.join(out_dir, "config.json")
    for stats in stats_generator:
        for k, v in stats.items():
            v['weights'] = os.path.basename(v['weights'])
        with open(config_out, "w") as f:
            json.dump({**config, 'train_output': {
                'stats': {**load_stats, **stats},
                'source': config_path,
                'environment': dict(os.environ)
            }}, f, indent='\t')
        logging.info("Wrote output to " + config_out)
    latest_path = os.path.join("trainNN", "out", "latest")
    with contextlib.suppress(FileNotFoundError):
        os.remove(latest_path)
    os.symlink(version, latest_path)


if __name__ == "__main__":
    train()
