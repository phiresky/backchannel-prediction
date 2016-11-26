import swda as swda
import re

corpus = swda.CorpusReader('swda')

utts = {}
for trans in corpus.iter_transcripts():
    for utt in trans.utterances:
        tag = utt.damsl_act_tag()
        if tag[0] == 'b':
            txt = re.sub("[,.#]", "", utt.text[:-1].strip()).strip()
            if (txt, tag) not in utts:
                utts[txt, tag] = 1
            else:
                utts[txt, tag] += 1

perc = 0
total = sum(utts.values())
count = 0
print("aggregated\tself\ttext\tcategory")
for k in sorted(utts, key=lambda x: -utts[x]):
    count += utts[k]
    print("\n".join(["{:.2f}%\t{:.2f}%\t{}\t{}\t{}".format((float(count) / total) * 100, float(utts[k]) / total * 100,
                                                       utts[k], k[0], k[1])]))
