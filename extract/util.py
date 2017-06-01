import functools
import json
import os
import pickle
from collections import OrderedDict
from typing import List, Tuple, Iterator
import hashlib
import json
import inspect

import logging


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.keys())))


@functools.lru_cache(maxsize=8)
def load_config(path):
    with open(path) as config_file:
        try:
            return json.load(config_file, object_pairs_hook=OrderedDict)
        except Exception as e:
            print(f"error loading {path}: {e}")
            raise e


def batch_list(list: List, n: int, include_last_partial: bool):
    l = len(list)
    for ndx in range(0, l, n):
        if not include_last_partial and ndx + n > l:
            continue
        yield ndx, list[ndx:min(ndx + n, l)]


def windowed_indices(total_frames: int, context_frames: int, context_stride: int):
    max_start_index = total_frames - context_frames * context_stride
    for i in range(0, max_start_index + 1):
        yield range(i, i + context_frames * context_stride, context_stride)


def invert_channel(channel: str):
    return dict(A="B", B="A")[channel]


def filter_ranges(numbers: List[float], ranges: List[Tuple[float, float]]):
    inx = 0
    if len(numbers) == 0:
        return
    for start, end in ranges:
        while numbers[inx] < start:
            inx += 1
            if inx >= len(numbers):
                return
        while numbers[inx] <= end:
            yield numbers[inx]
            inx += 1
            if inx >= len(numbers):
                return


def get_talking_segments(reader, convid: str, invert: bool, min_talk_len=None) -> Iterator[
    Tuple[float, float]]:
    talk_start = 0
    talking = False
    utts = list(reader.get_utterances(convid))
    for index, (utt, uttInfo) in enumerate(utts):
        is_bc = reader.is_backchannel(uttInfo, index, utts)
        is_empty = len(reader.noise_filter(uttInfo['text'])) == 0
        if (is_bc or is_empty) != invert:
            if talking:
                talking = False
                talk_end = float(uttInfo['from'])
                if min_talk_len is None or talk_end - talk_start >= min_talk_len:
                    yield talk_start, talk_end
        else:
            if not talking:
                talking = True
                talk_start = float(uttInfo['from'])
    if talking:
        talk_end = float(utts[-1][1]['to'])
        if min_talk_len is None or talk_end - talk_start >= min_talk_len:
            yield talk_start, talk_end


def get_monologuing_segments(reader, convid: str, min_talk_len=None) -> Iterator[Tuple[float, float]]:
    bc_convid = convid[:-1] + dict(A="B", B="A")[convid[-1]]
    talking_segs = get_talking_segments(reader, convid, False)
    listening_segs = get_talking_segments(reader, bc_convid, True)
    all = []
    for start, end in talking_segs:
        all.append((start, "start", "talking"))
        all.append((end, "end", "talking"))
    for start, end in listening_segs:
        all.append((start, "start", "listening"))
        all.append((end, "end", "listening"))
    all.sort(key=lambda x: x[0])
    talking = False
    listening = False
    monologuing = False
    monologuing_start = 0
    for time, type, mode in all:
        is_starting = type == "start"
        if mode == "talking":
            talking = is_starting
        if mode == "listening":
            listening = is_starting
        if talking and listening:
            if not monologuing:
                monologuing = True
                monologuing_start = time
        else:
            if monologuing:
                monologuing = False
                monologuing_end = time
                if min_talk_len is None or monologuing_end - monologuing_start >= min_talk_len:
                    yield monologuing_start, monologuing_end
    if monologuing:
        monologuing_end = float(all[-1][0])
        if min_talk_len is None or monologuing_end - monologuing_start >= min_talk_len:
            yield monologuing_start, monologuing_end


@functools.lru_cache()
def getsourcelines_cached(f):
    return inspect.getsourcelines(f)[0]


# cache file to disc. todo: locking
def DiskCache(f):
    @functools.wraps(f)
    def wrap(*args, **kwargs):
        args_to_pickle = list(args)
        meta = dict(fnname=f.__name__, fnsource=getsourcelines_cached(f), args=args_to_pickle, kwargs=kwargs)
        meta_json = json.dumps(meta, sort_keys=True, indent='\t').encode('ascii')
        digest = hashlib.sha256(meta_json).hexdigest()
        path = os.path.join('data/cache', digest[0:2], digest[2:] + ".pickle")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as file:
                    return pickle.load(file)
            except Exception as e:
                logging.warning(f"could not read cached file {path} ({repr(e)}, {e}), recomputing")
        val = f(*args, **kwargs)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", 'wb') as file:
            pickle.dump(val, file, protocol=pickle.HIGHEST_PROTOCOL)
        if os.path.isfile(path + ".part"):
            os.rename(path + ".part", path)
            with open(path + '.meta.json', 'wb') as file:
                file.write(meta_json)
        else:
            logging.warning(f"could not find file {path}.part after writing")
        return val

    return wrap


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def cache_usage():
    from glob import glob
    from tqdm import tqdm
    from os.path import join
    from pprint import pprint
    from time import perf_counter
    import json
    dir = "data/cache"
    stats = {}
    last = perf_counter()
    total = 256
    cur = 0

    def print_stats(factor=1):
        it = list(stats.items())
        it.sort(key=lambda a: a[1])
        total = 0
        for fnname, size in it:
            total += size
            print(f"{sizeof_fmt(size * factor): >10} {fnname}")
        print(f"{sizeof_fmt(total * factor): >10} total")

    for dir in tqdm(glob(join(dir, "??"))):
        cur += 1
        for metaf in glob(join(dir, "*.meta.json")):
            fname = os.path.splitext(metaf)[0]
            fname = os.path.splitext(fname)[0]
            with open(metaf) as f:
                meta = json.load(f)
            fnname = meta['fnname']
            stats.setdefault(fnname, 0)
            stats[fnname] += os.stat(fname).st_size
        now = perf_counter()
        if perf_counter() - last > 10:
            print("interpolated stats:")
            print_stats(total / cur)
            last = now
    print_stats()


if __name__ == '__main__':
    cache_usage()
