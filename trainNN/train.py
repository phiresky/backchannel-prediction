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
from extract_pfiles_python.util import load_config, batch_list, windowed_indices
from trainNN import train_func
from . import network_model
from extract_pfiles_python import readDB
import functools
import time
from tqdm import tqdm
from joblib import Parallel, delayed

NUM_EPOCHS = 10000

reader = None
backchannels = None
config_path = None


def extract(utterance: Tuple[str, bool]):
    return readDB.extract(config_path)[utterance]


def extract_batch(batch: List[Tuple[Tuple[str, bool], List[int]]], context_frames: int, input_dim: int,
                  output_all: bool):
    inputs = np.empty((len(batch), context_frames, input_dim), dtype='float32')
    outputs = np.empty((len(batch),), dtype='int32') if not output_all else np.empty((len(batch), context_frames, 1),
                                                                                     dtype='int32')
    for index, (utterance_id, indices) in enumerate(batch):
        cur_inputs, cur_outputs = extract(utterance_id)
        inputs[index] = cur_inputs[indices]
        outputs[index] = cur_outputs[indices][0] if not output_all else cur_outputs[indices]
    return inputs, outputs


def iterate_minibatches(train_config, all_elements, output_all, random=True):
    batchsize = train_config['batch_size']
    context_frames = train_config['context_frames']
    input_dim = train_config['input_dim']

    if random:
        numpy.random.shuffle(all_elements)
    for _, batch in batch_list(all_elements, batchsize, include_last_partial=False):
        yield extract_batch(batch, context_frames, input_dim, output_all)


def load_numpy_file(fname):
    logging.info("loading numpy file " + fname)
    data = numpy.load(fname)['data']
    return data


def train():
    global reader
    global backchannels
    global config_path

    config_path = sys.argv[1]
    config = load_config(config_path)
    version = subprocess.check_output("git describe --dirty", shell=True).decode('ascii').strip()

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
    reader = readDB.DBReader(config, config_path)
    train_config = config['train_config']
    context_stride = train_config['context_stride']
    batch_size = train_config['batch_size']
    context_frames = int(train_config['context_ms'] / 10 / context_stride)
    train_config['context_frames'] = context_frames

    dir = os.path.dirname(config_path)
    convos = readDB.read_conversations(config)
    gaussian = train_config['gaussian']
    if gaussian:
        pass
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
            backchannels = list(readDB.all_uttids(reader, convos[t]))
            input_dim = extract(backchannels[0])[0].shape[1]
            logging.debug(f"set input dim to {input_dim}")
            train_config['input_dim'] = input_dim
            out_all = {'all': True, 'single': False}[train_config['output_type']]
            logging.debug(f"input dim = {input_dim}")
            context_stride = train_config['context_stride']
            context_length = int(train_config['context_ms'] / 10 / context_stride)
            sequence_length = int((reader.method['nbc'][1] - reader.method['nbc'][0]) * 1000 / 10)
            inner_indices = windowed_indices(sequence_length, context_length, context_stride)
            all_elements = list(itertools.product(backchannels, inner_indices))
            batchers[t] = partial(iterate_minibatches, train_config, all_elements, out_all)
            before = time.perf_counter()
            before_cpu = time.process_time()
            logging.debug("loading data into ram")
            i = 0
            for backchannel in tqdm(backchannels):
                extract(backchannel)
            logging.debug(
                f"loading data took {time.perf_counter () - before:.3f}s (cpu: {time.process_time()-before_cpu:.3f}s")

    create_network = getattr(network_model, train_config['model_function'])
    model = create_network(train_config)

    stats_generator = train_func.train_network(
        network=model['output_layer'],
        scheduling_method=None,
        resume=train_config['resume_parameters'],
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
                'stats': stats,
                'source': config_path,
            }}, f, indent='\t')
        logging.info("Wrote output to " + config_out)
    latest_path = os.path.join("trainNN", "out", "latest")
    with contextlib.suppress(FileNotFoundError):
        os.remove(latest_path)
    os.symlink(version, latest_path)


if __name__ == "__main__":
    train()
