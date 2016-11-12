from jrtk.features import PitchTracker
from jrtk.preprocessing import AbstractStep, NumFeature, FeatureExtractor
from typing import Dict


class Step(AbstractStep):
    def __init__(self, config):
        super().__init__(config)

    def eval(self, featExtractor: FeatureExtractor, spkInfo, uttInfo, feats: Dict[str, NumFeature]):
        tracker = PitchTracker()
        [pitcha, pitchb] = [feats[which].applyPitchTracker(tracker, window='32ms')
                            for which in ('adca', 'adcb')]  # type: NumFeature

        tmpfeata = pitcha.merge(feats['powera'])  # type: NumFeature
        tmpfeatb = pitchb.merge(feats['powerb'])  # type: NumFeature

        feata = tmpfeata.adjacent(featExtractor.config['delta'])
        featb = tmpfeatb.adjacent(featExtractor.config['delta'])
        return {
            'pitcha': pitcha,
            'pitchb': pitchb,
            'feata': feata,
            'featb': featb
        }