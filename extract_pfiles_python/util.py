import functools
import json
from collections import OrderedDict
from typing import List


@functools.lru_cache(maxsize=8)
def load_config(path):
    with open(path) as config_file:
        return json.load(config_file, object_pairs_hook=OrderedDict)


def batch_list(list: List, n: int, include_last_partial: bool):
    l = len(list)
    for ndx in range(0, l, n):
        if not include_last_partial and ndx + n > l:
            continue
        yield ndx, list[ndx:min(ndx + n, l)]


def windowed_indices(total_frames: int, context_frames: int, context_stride: int):
    max_start_index = total_frames - context_frames * context_stride
    for i in range(0, max_start_index):
        yield range(i, i + context_frames * context_stride, context_stride)


def invert_channel(channel: str):
    return dict(A="B", B="A")[channel]
