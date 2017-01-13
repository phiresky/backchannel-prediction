# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/featAccess
# (sha1sum c284a64313b39c81171a3a8de06383171e5731e)
# on 2016-11-06

from jrtk.preprocessing import FeatureExtractor, AbstractStep, NumFeature
from jrtk.features import Filter, PitchTracker, FeatureType, FeatureSet
import os
import numpy as np
import soundfile as sf
import functools
import pickle
import logging
from typing import Iterable
from trainNN.evaluate import get_network_outputter


def NumFeature_to_dict(n: NumFeature):
    return {
        'samplingRate': n.samplingRate,
        'data': n,
        'shift': n.shift
    }


def NumFeature_from_dict(d: dict):
    return NumFeature(d['data'], samplingRate=d['samplingRate'], shift=d['shift'])


def NumFeatureCache(f):
    @functools.wraps(f)
    def wrap(featuresInstance, *args, **kwargs):
        key = ".".join([featuresInstance.config_path.replace("/", "_"), f.__name__, *[str(arg) for arg in args], *["{}={}".format(arg, val) for arg, val in kwargs.items()]])
        path = os.path.join(featuresInstance.config['paths']['cacheDirectory'], key)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return NumFeature_from_dict(pickle.load(file))
        else:
            logging.info("cache miss: " + path)
            val = f(featuresInstance, *args, **kwargs)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", 'wb') as file:
                pickle.dump(NumFeature_to_dict(val), file, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(path + ".part", path)
            return val

    return wrap


filtr = Filter(-2, [1, 2, 3, 2, 1])


def filter_power(power: NumFeature) -> NumFeature:
    b = power.max() / 10 ** 4
    val = power + b
    for i in range(0, len(val)):
        if val[i] <= 0:
            val[i] = 1
    power = np.log10(val)
    power = power.applyFilter(filtr)
    power = power.applyFilter(filtr)
    power = power.normalize(min=-1, max=1)
    return power


tracker = PitchTracker()


def power_transform(adc: NumFeature, sample_window_ms: float) -> NumFeature:
    return filter_power(adc.adc2pow("{}ms".format(sample_window_ms)))


def pitch_transform(adc: NumFeature, sample_window_ms: float) -> NumFeature:
    return adc.applyPitchTracker(tracker, window="{}ms".format(sample_window_ms)).normalize(min=-1, max=1)


def adjacent(feat: NumFeature, offsets: Iterable[int]):
    offsets = list(offsets)
    offset_count = len(offsets)
    if feat.typ != FeatureType.FMatrix:
        raise Exception("only works for FMatrix")
    (frame_count, feature_dimension) = feat.shape
    out_feat = np.zeros((frame_count, offset_count * feature_dimension), dtype='float32')
    offsets = np.array(offsets)
    out_offsets = feature_dimension * np.array(range(len(offsets) + 1))
    frame_indices = np.array(range(frame_count))
    for column, offset in enumerate(offsets):
        column_offsets = np.clip(frame_indices + offset, 0, frame_count - 1)
        out_feat[:, out_offsets[column]:out_offsets[column + 1]] = feat[column_offsets]
    return NumFeature(out_feat, shift=feat.shift, samplingRate=feat.samplingRate)


class Features:
    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.config_path = config_path
        self.sample_window_ms = config['extract_config']['sample_window_ms']  # type: float

    @functools.lru_cache(maxsize=32)
    @NumFeatureCache
    def get_adc(self, convid: str) -> NumFeature:
        conv, channel = convid.split("-")
        adcfile = os.path.join(self.config['paths']['adc'], conv + ".wav")
        if not os.path.exists(adcfile):
            raise Exception("cannot find adc for {}, file {} does not exist".format(conv, adcfile))
        res = readAudioFile(adcfile)
        [ADC0A, ADC0B] = [x.substractMean() for x in res]  # type: NumFeature
        if channel == "A":
            return ADC0A
        if channel == "B":
            return ADC0B

    @functools.lru_cache(maxsize=32)
    @NumFeatureCache
    def get_power(self, convid: str) -> NumFeature:
        return power_transform(self.get_adc(convid), self.sample_window_ms)

    @functools.lru_cache(maxsize=32)
    @NumFeatureCache
    def get_pitch(self, convid: str) -> NumFeature:
        return pitch_transform(self.get_adc(convid), self.sample_window_ms)

    @functools.lru_cache(maxsize=16)
    def get_combined_feat(self, convid: str) -> NumFeature:
        pitch = self.get_pitch(convid)
        power = self.get_power(convid)
        ex_config = self.config['extract_config']
        context_ms = ex_config['context_ms']
        ms_shift = power.shift
        context = int(context_ms / ms_shift)
        stride = ex_config['context_stride']
        online = ex_config['online'] if 'online' in ex_config else False
        if not online:
            offsets = range(-context // 2, context // 2, stride)
        else:
            offsets = range(stride - context, 1, stride)
        return adjacent(pitch.merge(power), offsets)

    @functools.lru_cache(maxsize=2)
    @NumFeatureCache
    def get_net_output(self, convid: str, epoch: str, smooth: bool):
        layers, fn = get_network_outputter(self.config_path, epoch)
        input = self.get_combined_feat(convid)
        batchsize, *dims = layers[0].shape
        framecount, *_ = input.shape
        shape = (framecount, *dims)
        output = fn(input.reshape(shape))
        if output.shape[1] == 1:
            feature = NumFeature(output)
        else:
            feature = NumFeature(output[:, [1]])
        if smooth:
            import scipy
            res = scipy.ndimage.filters.gaussian_filter1d(feature, 300 / feature.shift, axis=0)
            return NumFeature(res)
        else:
            return feature

    def sample_index_to_time(self, feat: NumFeature, sample_index):
        if feat.typ == FeatureType.FMatrix:
            return (self.sample_window_ms / 2 + sample_index * feat.shift) / 1000
        if feat.typ == FeatureType.SVector:
            return sample_index / (feat.samplingRate * 1000)
        raise Exception("Unknown feature type")

    def time_to_sample_index(self, feat: NumFeature, time_sec):
        if feat.typ == FeatureType.FMatrix:
            return int(round((1000 * time_sec - self.sample_window_ms / 2) / feat.shift))
        if feat.typ == FeatureType.SVector:
            return int(round(1000 * feat.samplingRate * time_sec))
        raise Exception("Unknown feature type")

    def cut_range(self, feat: NumFeature, from_time: float, to_time: float):
        if feat.typ == FeatureType.SVector:
            return feat[str(from_time) + "s":str(to_time) + "s"]
        else:
            return feat[self.time_to_sample_index(feat, from_time): self.time_to_sample_index(feat, to_time)]


fs = FeatureSet()


@functools.lru_cache(maxsize=1)
def readAudioFile(filename: str, dtype='int16', **kwargs):
    """Thin wrapper around the soundfile.read() method. Arguments are passed through, data read is returned as NumFeature.

    For a complete list of arguments and description see http://pysoundfile.readthedocs.org/en/0.8.1/#module-soundfile

    Returns:
        Single NumFeature if the audio file had only 1 channel, otherwise a list of NumFeatures.
    """
    logging.debug("parsing audio file " + filename)
    data, samplingRate = sf.read(filename, dtype=dtype, **kwargs)

    if data.ndim == 2:
        # multi channel, each column is a channel
        return [NumFeature(col, samplingRate=samplingRate / 1000, shift=0) for col in data.T]
    return NumFeature(data, featureSet=fs, samplingRate=samplingRate / 1000, shift=0)


def test():
    f = Features(dict(extract_config=dict(sample_window_ms=32)))
    arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype='float32')
    feat = NumFeature(arr)
    print(feat)
    print(feat.adjacent(1))
    print(f.adjacent(feat, [-1, 0, 1]))
    print(f.adjacent(feat, range(-5, 1)))


if __name__ == '__main__':
    test()
