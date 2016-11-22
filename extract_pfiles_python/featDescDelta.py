# originally ported from
# /project/dialog/backchanneler/workspace/lars_context/extract_pfiles/featDescDelta
# (sha1sum 61e11d26f93f8a4792dffdaf49440e536fb70217)
# on 2016-11-06

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
        pitcha = pitcha / 200
        pitchb = pitchb / 200
        tmpfeata = pitcha.merge(feats['powera'])  # type: NumFeature
        tmpfeatb = pitchb.merge(feats['powerb'])  # type: NumFeature

        feata = tmpfeata.adjacent(featExtractor.config['context'])
        featb = tmpfeatb.adjacent(featExtractor.config['context'])
        return {
            'pitcha': pitcha,
            'pitchb': pitchb,
            'feata': feata,
            'featb': featb
        }
