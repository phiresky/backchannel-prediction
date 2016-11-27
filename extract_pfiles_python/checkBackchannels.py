from . import readDB
import sys


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


config = readDB.load_config(sys.argv[1])


def check_transcript_differences():
    config['extract_config']['useOriginalDB'] = False
    isl_reader = readDB.DBReader(config)

    list1 = {(spkr, bcInfo['from'], bcInfo['to']): bcInfo for spkr in isl_reader.spkDB if spkr[0:2] == 'sw' for
             (_, bcInfo) in isl_reader.get_backchannels(list(isl_reader.get_utterances(spkr)))}

    config['extract_config']['useOriginalDB'] = True
    orig_reader = readDB.DBReader(config)

    list2 = {(spkr, bcInfo['from'], bcInfo['to']): bcInfo for spkr in orig_reader.spkDB if spkr[0:2] == 'sw' for
             (_, bcInfo) in orig_reader.get_backchannels(list(orig_reader.get_utterances(spkr)))}

    print("diffing")
    for bc in list1:
        if bc not in list2:
            print("not in orig: {}".format(list1[bc]))
            print("there is: {}".format(
                [uttInfo for _, uttInfo in orig_reader.get_utterances(bc[0]) if uttInfo['from'] == bc[1]]))
            print("before is: {}".format(
                [uttInfo for _, uttInfo in orig_reader.get_utterances(bc[0]) if uttInfo['to'] == bc[1]]))
            print()
    for bc in list2:
        if bc not in list1:
            print("not in isl: {}".format(list2[bc]))
            print("there is: {}".format(
                [uttInfo for _, uttInfo in isl_reader.get_utterances(bc[0]) if uttInfo['from'] == bc[1]]))
            print("before is: {}".format(
                [uttInfo for _, uttInfo in isl_reader.get_utterances(bc[0]) if uttInfo['to'] == bc[1]]))
            print()
    print("with isl db: " + str(len(list1)))
    print("with orig db: " + str(len(list2)))


def count_utterances(exclude_backchannels=False):
    reader = readDB.DBReader(config)
    utts = {}
    spkrcount = 0
    for spkr in reader.spkDB:
        if spkr[0:3] == "en_":
            continue
        spkrcount += 1
        for (utt, uttInfo) in reader.get_utterances(spkr):
            txt = uttInfo['text']
            if txt not in utts:
                utts[txt] = 1
            else:
                utts[txt] += 1

    print("spkrcount={}".format(spkrcount))
    perc = 0
    total = sum(utts.values())
    count = 0
    print("bc\taggregated\tself\tcount\ttext")

    for k in sorted(utts, key=lambda x: -utts[x]):
        if utts[k] < 10:
            break
        count += utts[k]
        is_backchannel = reader.noise_filter(k.lower()) in reader.backchannels
        if exclude_backchannels and is_backchannel:
            continue
        print("\n".join(["[{}BC]\t{:.2f}%\t{:.2f}%\t{}\t{}".format(" " if is_backchannel else "N",
                                                                   (float(count) / total) * 100,
                                                                   float(utts[k]) / total * 100,
                                                                   utts[k], k)]))


def count_backchannels():
    extract_config = config['extract_config']
    context = extract_config['context']

    with readDB.DBReader(config) as reader:
        for setname, path in config['paths']['conversations'].items():
            with open(path) as f:
                convIDs = set([line.strip() for line in f.readlines()])
            print("bc counts for {}: {}".format(setname, reader.count_total(convIDs)))


def sanity():
    """check if spkDB and uttDB have the same utterances"""
    reader = readDB.DBReader(config)
    utts = list()
    for spk in reader.spkDB:
        x = reader.spkDB[spk]
        utts += x['segs'].strip().split(" ")

    utts2 = list()
    x = list(reader.uttDB)
    for utt in reader.uttDB:
        x = reader.uttDB[utt]
        utts2.append(utt)

    print(utts == utts2)


# check_transcript_differences()
count_backchannels()
