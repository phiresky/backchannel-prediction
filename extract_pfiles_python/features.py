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
    def wrap(featuresInstance, convid):
        path = os.path.join(featuresInstance.config['paths']['cacheDirectory'], convid + "." + f.__name__)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return NumFeature_from_dict(pickle.load(file))
        else:
            logging.debug("cache miss: " + path)
            val = f(featuresInstance, convid)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", 'wb') as file:
                pickle.dump(NumFeature_to_dict(val), file, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(path + ".part", path)
            return val

    return wrap


class Features:
    filtr = Filter(-2, [1, 2, 3, 2, 1])

    def __init__(self, config):
        self.config = config
        self.sample_window_ms = config['extract_config']['sample_window_ms']  # type: float
        self.tracker = PitchTracker()

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
        powerraw = self.get_adc(convid).adc2pow("{}ms".format(self.sample_window_ms))
        return self.filter_power(powerraw)

    @functools.lru_cache(maxsize=32)
    @NumFeatureCache
    def get_pitch(self, convid: str) -> NumFeature:
        return self.get_adc(convid).applyPitchTracker(self.tracker,
                                                      window="{}ms".format(self.sample_window_ms)).normalize(min=-1,
                                                                                                             max=1)

    def get_combined_feat(self, convid: str) -> NumFeature:
        pitch = self.get_pitch(convid)
        power = self.get_power(convid)
        return pitch.merge(power).adjacent(self.config['extract_config']['context'])

    def filter_power(self, power: NumFeature) -> NumFeature:
        b = power.max() / 10 ** 4
        val = power + b
        for i in range(0, len(val)):
            if val[i] <= 0:
                val[i] = 1
        power = np.log10(val)
        power = power.applyFilter(self.filtr)
        power = power.applyFilter(self.filtr)
        power = power.normalize(min=-1, max=1)
        return power

    def sample_index_to_time(self, feat: NumFeature, sample_index):
        if feat.typ != FeatureType.FMatrix:
            raise Exception("only for extracted features")
        return (self.sample_window_ms / 2 + sample_index * feat.shift) / 1000

    def time_to_sample_index(self, feat: NumFeature, time_sec):
        if feat.typ != FeatureType.FMatrix:
            raise Exception("only for extracted features")
        return int(round((1000 * time_sec - self.sample_window_ms / 2) / feat.shift))

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
