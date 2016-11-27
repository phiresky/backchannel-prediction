from . import readDB


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


config = readDB.load_config("extract_pfiles_python/config.json")
readDB.load_backchannels(config['paths']['backchannels'])
spkDB, uttDB = readDB.load_db(config['paths'])

utts = {}
spkrcount = 0
for spkr in spkDB:
    if spkr[0:3] == "en_":
        continue
    spkrcount += 1
    for utt in readDB.getUtterances(spkDB, spkr):
        uttInfo = uttDB[utt]
        txt = uttInfo['text']
        if txt not in utts:
            utts[txt] = 1
        else:
            utts[txt] += 1

print("spkrcount={}".format(spkrcount))
perc = 0
total = sum(utts.values())
count = 0
print("aggregated\tself\tcount\ttext")

for k in sorted(utts, key=lambda x: -utts[x]):
    if utts[k] < 10:
        break
    count += utts[k]
    if k.lower() in readDB.backchannels:
        continue;
    print("\n".join(["{:.2f}%\t{:.2f}%\t{}\t{}".format((float(count) / total) * 100, float(utts[k]) / total * 100,
                                                       utts[k], k)]))
