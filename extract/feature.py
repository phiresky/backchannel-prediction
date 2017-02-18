import numpy as np
import functools
import logging
from jrtk.preprocessing import NumFeature, AbstractStep
from jrtk.features import Filter, PitchTracker, FeatureType, FeatureSet
from typing import Union, List

tracker = PitchTracker()


# see https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class Feature(np.ndarray):
    frame_window_ms: int = None
    frame_shift_ms: int = None
    """
    the offset in milliseconds that is the center of the first frame
    for example, when extracting 32ms windows with 10ms shift the first frame will have an offset of 16ms
    """
    initial_frame_offset_ms: float = None

    def __new__(cls, input_array, infofrom: Union['Feature', dict]):
        obj = np.asarray(input_array).view(cls)
        if infofrom is None:
            raise Exception("did not supply info source")
        if not isinstance(infofrom, dict):
            infofrom = infofrom.get_info_dict()
        obj.add_info_from_dict(infofrom)
        return obj

    def add_info_from_dict(self, infofrom: dict):
        self.frame_window_ms = infofrom['frame_window_ms']
        self.frame_shift_ms = infofrom['frame_shift_ms']
        self.initial_frame_offset_ms = infofrom['initial_frame_offset_ms']

    def get_info_dict(self):
        return dict(frame_window_ms=self.frame_window_ms, frame_shift_ms=self.frame_shift_ms,
                    initial_frame_offset_ms=self.initial_frame_offset_ms)

    def __array_finalize__(self, obj):
        if not isinstance(obj, Feature):
            return
        self.frame_window_ms = obj.frame_window_ms
        self.frame_shift_ms = obj.frame_shift_ms
        self.initial_frame_offset_ms = obj.initial_frame_offset_ms

    # for pickling
    # http://stackoverflow.com/a/26599346/2639190
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Feature, self).__reduce__()
        info = self.get_info_dict()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (info,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        *state, info = state
        self.add_info_from_dict(info)  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Feature, self).__setstate__(state)

    def sample_index_to_time(self, sample_index: int) -> float:
        return (self.initial_frame_offset_ms + sample_index * self.frame_shift_ms) / 1000

    def time_to_sample_index(self, time_sec: float) -> int:
        return int(round((1000 * time_sec - self.initial_frame_offset_ms) / self.frame_shift_ms))

    def cut_by_time(self, from_time: float, to_time: float):
        from_inx = self.time_to_sample_index(from_time)
        to_inx = self.time_to_sample_index(to_time)
        actual_from_time = self.sample_index_to_time(from_inx)
        cut = self[from_inx:to_inx]
        mytime = self.initial_frame_offset_ms
        cut.initial_frame_offset_ms = (from_time - actual_from_time) * 1000
        assert mytime == self.initial_frame_offset_ms
        return cut


melMatrix = None


def ms(ms: int):
    return f"{ms}ms"


class Audio(np.ndarray):
    sample_rate_hz: int = None

    def __new__(cls, input_array, sample_rate_hz: int):
        obj = np.asarray(input_array).view(cls)
        obj.sample_rate_hz = sample_rate_hz
        return obj

    # for pickling
    # http://stackoverflow.com/a/26599346/2639190
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Audio, self).__reduce__()
        info = dict(sample_rate_hz=self.sample_rate_hz)
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (info,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        *state, info = state
        self.sample_rate_hz = info.get('sample_rate_hz', None)
        # Call the parent's __setstate__ with the other tuple elements.
        super(Audio, self).__setstate__(state)

    def __array_finalize__(self, obj):
        assert len(self.shape) == 1
        # assert self.dtype == np.int16

        if obj is None:
            return
        self.sample_rate_hz = getattr(obj, "sample_rate_hz", None)

    def sample_index_to_time(self, sample_index: int) -> float:
        return sample_index / self.sample_rate_hz

    def time_to_sample_index(self, time_sec: int) -> int:
        return int(round(self.sample_rate_hz * time_sec))

    def to_num_feature(self):
        return NumFeature(self, samplingRate=self.sample_rate_hz / 1000)

    def get_power(self, frame_window_ms: int, frame_shift_ms: int = 10):
        pow = self.to_num_feature().adc2pow(window=ms(frame_window_ms), shift=ms(frame_shift_ms))
        assert pow.shift == frame_shift_ms
        f = Feature(pow, infofrom=dict(frame_window_ms=frame_window_ms, frame_shift_ms=frame_shift_ms,
                                       initial_frame_offset_ms=frame_window_ms // 2))
        return f

    def get_intonation(self, frame_window_ms: int, frame_shift_ms: int = 10) -> Feature:
        ffv = self.to_num_feature().intonation(window=ms(frame_window_ms), shift=ms(frame_shift_ms))
        return Feature(ffv, infofrom=dict(frame_window_ms=frame_window_ms, frame_shift_ms=frame_shift_ms,
                                          initial_frame_offset_ms=frame_window_ms // 2))

    def get_pitch(self, frame_window_ms: int, frame_shift_ms: int = 10) -> Feature:
        pitch = self.to_num_feature().applyPitchTracker(tracker, window=ms(frame_window_ms),
                                                        shift=ms(frame_shift_ms)).normalize(min=-1, max=1)
        return Feature(pitch, infofrom=dict(frame_window_ms=frame_window_ms, frame_shift_ms=frame_shift_ms,
                                            initial_frame_offset_ms=frame_window_ms // 2))

    def get_mfcc(self, frame_window_ms: int, frame_shift_ms: int = 10):
        # adapted from /project/iwslt2016/DE/sys2016/B-systems/B-02-samples-MFCC+MVDR+T/featDesc
        # https://bitbucket.org/jrtk/janus/wiki/Python/Preprocessing
        import scipy.fftpack
        import sklearn.preprocessing
        global melMatrix

        fft0 = self.to_num_feature().spectrum(window=ms(frame_window_ms), shift=ms(frame_shift_ms))
        fft = fft0.vtln(ratio=1.0, edge=0.8, mod='lin')
        if melMatrix is None:
            melMatrix = AbstractStep._getMelMatrix(None, filters=30, points=fft.shape[1], samplingRate=fft.samplingRate)
        mel = fft.filterbank(melMatrix)
        lMEL = np.log10(mel + 1)

        cepN = 20
        MFCC_MCEP = scipy.fftpack.dct(x=lMEL, type=2, n=cepN, norm="ortho")
        # normalize. warning: this is not actually possible in realtime because we don't yet know global min/max!
        MFCC_MCEP = sklearn.preprocessing.normalize(MFCC_MCEP, axis=0, norm='max')
        return Feature(MFCC_MCEP, infofrom=dict(frame_window_ms=frame_window_ms, frame_shift_ms=frame_shift_ms,
                                                initial_frame_offset_ms=frame_window_ms // 2))


def normalize(self, min, max):
    oldRange = self.max() - self.min()
    newRange = max - min
    if oldRange == 0:
        factor = 1
    else:
        factor = newRange / oldRange
    return (self - self.min()) * factor + min


def filter_power(_power: Feature) -> Feature:
    from scipy.signal import convolve
    b = _power.max() / 10 ** 4
    val = _power + b
    for i in range(0, len(val)):
        if val[i] <= 0:
            val[i] = 1
    power = np.log10(val)
    filter = np.array([1, 2, 3, 2, 1], dtype=np.float32).reshape((-1, 1))
    power = convolve(power, filter, mode='same')
    power = convolve(power, filter, mode='same')
    # normalize. warning: this is not actually possible in realtime because we don't yet know global min/max!
    power = normalize(power, min=-1, max=1)
    return Feature(power, infofrom=_power)


def filter_raw_power(_power: Feature) -> Feature:
    b = _power.max() / 10 ** 4
    val = _power + b
    for i in range(0, len(val)):
        if val[i] <= 0:
            val[i] = 1
    power = np.log10(val)
    return Feature(power, infofrom=_power)


@functools.lru_cache(maxsize=1)
def readAudioFile(filename: str, dtype='int16', **kwargs) -> List[Audio]:
    logging.debug("parsing audio file " + filename)
    import soundfile
    data, samplingRate = soundfile.read(filename, dtype=dtype, **kwargs)

    if data.ndim == 2:
        # multi channel, each column is a channel
        return [Audio(col, sample_rate_hz=samplingRate) for col in data.T]
    return [Audio(data, sample_rate_hz=samplingRate)]
