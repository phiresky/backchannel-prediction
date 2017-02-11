import functools
import json
from collections import OrderedDict
from typing import List, Tuple, Iterator


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
