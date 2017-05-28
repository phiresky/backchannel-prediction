# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/featAccess
# (sha1sum c284a64313b39c81171a3a8de06383171e5731e)
# on 2016-11-06


import os
import numpy as np
import soundfile as sf
import functools
import pickle
import logging
from typing import Iterable
from trainNN.evaluate import get_network_outputter

from tqdm import tqdm, trange
from .feature import Feature, Audio, filter_raw_power, filter_power, readAudioFile
from .util import DiskCache, load_config


def power_transform(adc: Audio, sample_window_ms: int) -> Feature:
    return filter_power(adc.get_power(sample_window_ms))


def raw_power_transform(adc: Audio, sample_window_ms: int) -> Feature:
    return filter_raw_power(adc.get_power(sample_window_ms))


def ffv_transform(adc: Audio, sample_window_ms: int) -> Feature:
    return adc.get_intonation(sample_window_ms)


def pitch_transform(adc: Audio, sample_window_ms: int) -> Feature:
    return adc.get_pitch(sample_window_ms)


def adjacent(feat: Feature, offsets: Iterable[int]):
    offsets = list(offsets)
    offset_count = len(offsets)
    (frame_count, feature_dimension) = feat.shape
    out_feat = np.zeros((frame_count, offset_count * feature_dimension), dtype='float32')
    offsets = np.array(offsets)
    out_offsets = feature_dimension * np.array(range(len(offsets) + 1))
    frame_indices = np.array(range(frame_count))
    for column, offset in enumerate(offsets):
        column_offsets = np.clip(frame_indices + offset, 0, frame_count - 1)
        out_feat[:, out_offsets[column]:out_offsets[column + 1]] = feat[column_offsets]
    return Feature(out_feat, infofrom=feat)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_adc(adc_path: str, convid: str) -> Audio:
    conv, channel = convid.split("-")
    adcfile = os.path.join(adc_path, conv + ".wav")
    if not os.path.exists(adcfile):
        raise Exception("cannot find adc for {}, file {} does not exist".format(conv, adcfile))
    res = readAudioFile(adcfile)
    [ADC0A, ADC0B] = [x - int(x.mean().round()) for x in res]
    return dict(A=ADC0A, B=ADC0B)[channel]


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_pitch(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return pitch_transform(pure_get_adc(adc_path, convid), sample_window_ms)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_power(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return power_transform(pure_get_adc(adc_path, convid), sample_window_ms)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_raw_power(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return raw_power_transform(pure_get_adc(adc_path, convid), sample_window_ms)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_ffv(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return ffv_transform(pure_get_adc(adc_path, convid), sample_window_ms)


@functools.lru_cache(maxsize=32)
def pure_cut_range(start_time: float, end_time: float, feat_fn: str, *args, **kwargs) -> Feature:
    return globals()[feat_fn](*args, **kwargs).cut_by_time(start_time, end_time)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_mfcc(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return pure_get_adc(adc_path, convid).get_mfcc(sample_window_ms)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_v1(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 5)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim10(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 10)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim15(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 15)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim20(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 20)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim30(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 30)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim30_4M(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 30, "4M")


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim30_4M_clean(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 30, "4Mclean")


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim40(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 40)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim41(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 41)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim50(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 50)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim75(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 75)


@functools.lru_cache(maxsize=32)
@DiskCache
def pure_get_word2vec_dim100(adc_path: str, sample_window_ms: int, convid: str) -> Feature:
    return get_word2vec(adc_path, sample_window_ms, convid, 100)


def get_word2vec(adc_path: str, sample_window_ms: int, convid: str, feat_dim: int, T="") -> Feature:
    from extract import readDB
    cp = "trainNN/out/v050-finunified-16-g1be124b-dirty:lstm-best-features-power,pitch,ffv,word2vec_dim30-slowbatch/config.json"
    model = readDB.word_to_vec(
        cp,
        dimension=feat_dim,
        T=T)
    # for dimensions
    pow = pure_get_power(adc_path, sample_window_ms, convid)
    frames, _ = pow.shape
    w2v = np.zeros((frames, feat_dim), dtype=np.float32)
    reader = readDB.loadDBReader(
        "configs/finunified/vary-features/lstm-best-features-power,ffv.json")  # exact config file is unimportant
    words = [(float(word['to']), reader.noise_filter(word['text'])) for word in
             readDB.get_all_nonsilent_words(reader, convid) if reader.noise_filter(word['text']) in model]
    inx = 0

    def inxtotime(sample_index):
        return (sample_window_ms / 2 + sample_index * pow.frame_shift_ms) / 1000

    for frame in range(frames):
        time = inxtotime(frame)
        if inx < len(words) - 1 and words[inx + 1][0] <= time:
            inx += 1
        w2v[frame] = model[words[inx][1]]
    return Feature(w2v, infofrom=pow)


@functools.lru_cache(maxsize=2)
@DiskCache
def pure_get_multidim_net_output(*, convid: str, epoch: str, config_path: str):
    layers, fn = get_network_outputter(config_path, epoch, batch_size=None)
    config = load_config(config_path)
    f = Features(config, config_path)
    input = f.get_combined_feature(convid)
    total_frames = input.shape[0]
    context_frames = config['train_config']['context_frames']
    context_stride = config['train_config']['context_stride']
    context_range = context_frames * context_stride

    def stacker():
        for frame in range(0, total_frames):
            if frame < context_range:
                # this is shitty / incorrect
                yield input[range(0, context_range, context_stride)]
            else:
                yield input[range(frame - context_range, frame, context_stride)]

    inp = np.stack(stacker())
    output = batched_eval(inp, fn, out_dim=config['train_config']['num_labels'])
    return Feature(output, infofrom=input)


def batched_eval(inputs, mapper, out_dim=1, batchsize=2000):
    from . import util
    total_frames = inputs.shape[0]
    output = np.zeros(shape=(total_frames, out_dim), dtype=np.float32)
    for ndx, batch in util.batch_list(inputs, batchsize, True):
        output[ndx:ndx + batch.shape[0]] = mapper(batch)
    return output


class Features:
    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.config_path = config_path
        self.sample_window_ms = config['extract_config']['sample_window_ms']  # type: int

    def get_adc(self, convid: str) -> Audio:
        return pure_get_adc(self.config['paths']['adc'], convid)

    def get_power(self, convid: str):
        return pure_get_power(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_raw_power(self, convid: str):
        return pure_get_raw_power(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_pitch(self, convid: str):
        return pure_get_pitch(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_ffv(self, convid: str):
        return pure_get_ffv(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_mfcc(self, convid: str):
        return pure_get_mfcc(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_word2vec_v1(self, convid: str):
        return pure_get_word2vec_v1(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_word2vec_dim10(self, convid: str):
        return pure_get_word2vec_dim10(self.config['paths']['adc'], self.sample_window_ms, convid)

    def get_combined_feature(self, convid: str, start_time: float = None, end_time: float = None):
        adc_path = self.config['paths']['adc']
        if start_time is None and end_time is None:
            feats = [globals()["pure_" + feature](adc_path, self.sample_window_ms, convid)
                     for feature in self.config['extract_config']['input_features']]
        else:
            feats = [pure_cut_range(start_time, end_time, "pure_" + feature, adc_path, self.sample_window_ms, convid)
                     for feature in self.config['extract_config']['input_features']]
        return Feature(np.concatenate(feats, axis=1), infofrom=feats[0])

    def get_multidim_net_output(self, convid: str, epoch: str):
        return pure_get_multidim_net_output(convid=convid, epoch=epoch, config_path=self.config_path)

    def smooth(self, convid: str, epoch: str, smoother: dict):
        x = self.get_multidim_net_output(convid, epoch)
        if smoother['type'].startswith("gauss"):
            import scipy.signal
            sigma = smoother['sigma_ms'] / x.frame_shift_ms
            if sigma < 1:
                return x
            cutoff = sigma * smoother['cutoff_sigma']
            # 4 x sigma contains 99.9 of values
            window = scipy.signal.gaussian(int(round(sigma * 2 * 4)), sigma).astype(np.float32)
            # cut off only on left side (after convolution this is the future)
            window = window[int(np.round(len(window) / 2 - cutoff)):]
            window = window / sum(window)
            x = Feature(np.array([scipy.signal.convolve(row, window)[:row.size] for row in x.T]).T, infofrom=x)
        elif smoother['type'].startswith("exponential"):
            factor = smoother['factor']
            facs = (factor * (1 - factor) ** np.arange(1000, dtype=np.float32))
            x = Feature(np.array([np.convolve(row, facs, mode='full')[:row.size] for row in x.T]).T, infofrom=x)
        elif smoother['type'] == 'kalman':
            # todo
            import pykalman
            kf = pykalman.KalmanFilter(n_dim_obs=x.shape[1])
            res, _ = kf.filter(x)  # features.gaussian_blur(x, 300)
            return None  # NumFeature(res.astype(np.float32))
        else:
            raise Exception(f"unknown method {smoother['type']}")
        return x
