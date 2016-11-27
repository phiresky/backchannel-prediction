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


def check_transcript_differences():
    readDB.load_backchannels(config['paths']['backchannels'])

    config['extract_config']['useOriginalDB'] = False
    spkDB1, uttDB1 = readDB.load_db(config)

    list1 = {(spkr, uttDB1[bc]['from'], uttDB1[bc]['to']): uttDB1[bc] for spkr in spkDB1 if spkr[0:2] == 'sw' for bc in
             readDB.getBackchannelIDs(uttDB1, list(readDB.get_utterance_ids(spkDB1, spkr)))}

    config['extract_config']['useOriginalDB'] = True
    spkDB2, uttDB2 = readDB.load_db(config)

    list2 = {(spkr, uttDB2[bc]['from'], uttDB2[bc]['to']): uttDB2[bc] for spkr in spkDB2 if spkr[0:2] == 'sw' for bc in
             readDB.getBackchannelIDs(uttDB2, list(readDB.get_utterance_ids(spkDB2, spkr)))}

    print("diffing")
    for bc in list1:
        if bc not in list2:
            print("not in orig: {}".format(list1[bc]))
            print("there is: {}".format(
                [uttDB2[utt] for utt in readDB.get_utterance_ids(spkDB2, bc[0]) if uttDB2[utt]['from'] == bc[1]]))
            print("before is: {}".format(
                [uttDB2[utt] for utt in readDB.get_utterance_ids(spkDB2, bc[0]) if uttDB2[utt]['to'] == bc[1]]))
            print()
    for bc in list2:
        if bc not in list1:
            print("not in isl: {}".format(list2[bc]))
            print("there is: {}".format(
                [uttDB1[utt] for utt in readDB.get_utterance_ids(spkDB1, bc[0]) if uttDB1[utt]['from'] == bc[1]]))
            print("before is: {}".format(
                [uttDB1[utt] for utt in readDB.get_utterance_ids(spkDB1, bc[0]) if uttDB1[utt]['to'] == bc[1]]))
            print()
    print("with isl db: " + str(len(list1)))
    print("with orig db: " + str(len(list2)))


def count_utterances():
    spkDB, uttDB = readDB.load_db(config)
    utts = {}
    spkrcount = 0
    for spkr in spkDB:
        if spkr[0:3] == "en_":
            continue
        spkrcount += 1
        for utt in readDB.get_utterance_ids(spkDB, spkr):
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

count_utterances()