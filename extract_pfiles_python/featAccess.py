# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/featAccess
# (sha1sum c284a64313b39c81171a3a8de06383171e5731e)
# on 2016-11-06

from jrtk.preprocessing import FeatureExtractor, AbstractStep, NumFeature
from jrtk.features import Filter
import os
import numpy as np
import soundfile as sf
import functools


class Step(AbstractStep):
    def __init__(self, config):
        super().__init__(config)
        self.filtr = Filter(-2, [1, 2, 3, 2, 1])

    def eval(self, featExtractor: FeatureExtractor, spkInfo, uttInfo, feats):
        conv = uttInfo['conv']
        adcfile = os.path.join(featExtractor.config['paths']['adc'], conv + ".wav")
        if not os.path.exists(adcfile):
            raise Exception("cannot find adc for {}, file {} does not exist".format(conv, adcfile))

        res = self.readAudioFile(featExtractor._featureSet, adcfile)
        [ADC0A, ADC0B] = res  # type: NumFeature

        adca = ADC0A[str(uttInfo['from']) + 's': str(uttInfo['to']) + 's'].substractMean()  # type: NumFeature
        adcb = ADC0B[str(uttInfo['from']) + 's': str(uttInfo['to']) + 's'].substractMean()  # type: NumFeature
        sample_window = str(featExtractor.config['extract_config']['sample_window_ms']) + "ms"
        powerrawa = adca.adc2pow(sample_window)
        powerrawb = adcb.adc2pow(sample_window)
        return {
            'adca': adca,
            'adcb': adcb,
            'powerrawa': powerrawa,
            'powerrawb': powerrawb,
            'powera': self.filterPower(powerrawa),
            'powerb': self.filterPower(powerrawb)
        }

    def filterPower(self, power: NumFeature) -> NumFeature:
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

    @functools.lru_cache(maxsize=10)
    def readAudioFile(self, featureSet, filename: str, *, dtype='int16', **kwargs):
        """Thin wrapper around the soundfile.read() method. Arguments are passed through, data read is returned as NumFeature.

        For a complete list of arguments and description see http://pysoundfile.readthedocs.org/en/0.8.1/#module-soundfile

        Returns:
            Single NumFeature if the audio file had only 1 channel, otherwise a list of NumFeatures.
        """
        data, samplingRate = sf.read(filename, dtype=dtype, **kwargs)

        if data.ndim == 2:
            # multi channel, each column is a channel
            return [NumFeature(col, featureSet=featureSet, samplingRate=samplingRate / 1000, shift=0) for col in data.T]
        return NumFeature(data, featureSet=featureSet, samplingRate=samplingRate / 1000, shift=0)
