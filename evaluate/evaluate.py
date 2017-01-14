import sys
import random
import soundfile
import os
import numpy as np
from jrtk.preprocessing import NumFeature
from typing import List, Tuple, Iterator
from extract_pfiles_python.readDB import load_config, DBReader, swap_speaker, read_conversations, load_config


def get_talking_segments(reader: DBReader, convid: str) -> Iterator[Tuple[float, float]]:
    talk_start = 0
    talking = False
    utts = list(reader.get_utterances(convid))
    for index, (utt, uttInfo) in enumerate(utts):
        is_bc = reader.is_backchannel(uttInfo, index, utts)
        is_empty = len(reader.noise_filter(uttInfo['text'])) == 0
        if is_bc or is_empty:
            if talking:
                talking = False
                yield talk_start, float(uttInfo['from'])
        else:
            if not talking:
                talking = True
                talk_start = float(uttInfo['from'])


def get_bc_samples(reader: DBReader, sampletrack="sw4687-B"):
    sampletrack_audio = reader.features.get_adc(sampletrack)
    bcs = reader.get_backchannels(list(reader.get_utterances(sampletrack)))
    for bc_id, bc_info in bcs:
        bc_start_time = float(bc_info['from'])
        bc_audio = reader.features.cut_range(sampletrack_audio, bc_start_time, float(bc_info['to']))
        bc_real_start_time = reader.getBcRealStartTime(bc_id)
        bc_start_offset = bc_real_start_time - bc_start_time
        yield bc_start_offset, bc_audio


# the word-aligned beginning of the bc is predicted
def predict_bcs(reader: DBReader, smoothed_net_output: NumFeature, threshold: float):
    for start, end in get_larger_threshold(smoothed_net_output, reader, threshold):
        peak_prediction_s = reader.get_max_time(smoothed_net_output, start, end) - reader.BCcenter
        yield peak_prediction_s


def get_bc_audio(smoothed_net_output: NumFeature, reader: DBReader,
                 bcs: List[Tuple[int, NumFeature]]):
    total_length_s = reader.features.sample_index_to_time(smoothed_net_output, smoothed_net_output.shape[0])
    total_length_audio_index = reader.features.time_to_sample_index(bcs[0][1], total_length_s)
    output_audio = NumFeature(np.zeros(total_length_audio_index, dtype='int16'),
                              samplingRate=bcs[0][1].samplingRate)

    for peak_s in predict_bcs(reader, smoothed_net_output, threshold=0.6):
        bc_start_offset, bc_audio = random.choice(bcs)
        audio_len_samples = bc_audio.shape[0]
        # audio_len_s = reader.features.sample_index_to_time(bc_audio, audio_len_samples)
        start_s = peak_s - bc_start_offset
        start_index = reader.features.time_to_sample_index(bc_audio, start_s)
        if start_index < 0:
            continue
        if start_index + audio_len_samples > output_audio.shape[0]:
            audio_len_samples = output_audio.shape[0] - start_index
        output_audio[start_index:start_index + audio_len_samples] += bc_audio[0: audio_len_samples]
    return output_audio


def get_larger_threshold(feat: NumFeature, reader: DBReader, threshold=0.5):
    begin = None
    for index, [sample] in enumerate(feat):
        if sample >= threshold and begin is None:
            begin = index
        elif sample < threshold and begin is not None:
            yield reader.features.sample_index_to_time(feat, begin), reader.features.sample_index_to_time(feat, index)
            begin = None


def normalize_audio(sampletrack_audio):
    max_amplitude = max(abs(sampletrack_audio.max()), abs(sampletrack_audio.min()))
    multi = 32767 / max_amplitude
    return (sampletrack_audio * multi).astype("int16")


# is this nearer than that from other?
def nearer(this: float, that: float, other):
    return abs(this - other) < abs(that - other)


def bc_is_within_margin_of_error(predicted: float, correct: float):
    return abs(predicted - correct) < 0.5


def evaluate_convs(reader: DBReader, convs: List[str]):
    selected_count = 0
    relevant_count = 0
    true_positives_count = 0
    false_positives_count = 0
    false_negatives_count = 0

    for conv in convs:
        for channel in ["A", "B"]:
            convid = "{}-{}".format(conv, channel)
            bc_convid = swap_speaker(convid)
            correct_bcs = [reader.getBcRealStartTime(utt) for utt, uttInfo in
                           reader.get_backchannels(list(reader.get_utterances(bc_convid)))]
            predicted_bcs = list(predict_bcs(reader, reader.features.get_net_output(convid, "best", smooth=True),
                                             threshold=0.6))
            predicted_count = len(predicted_bcs)
            predicted_inx = 0
            for correct_bc in correct_bcs:
                while predicted_inx < predicted_count - 1 and nearer(predicted_bcs[predicted_inx + 1],
                                                                     predicted_bcs[predicted_inx], correct_bc):
                    predicted_inx += 1
                if bc_is_within_margin_of_error(predicted_bcs[predicted_inx], correct_bc):
                    predicted_bcs[predicted_inx] = correct_bc

            # https://www.wikiwand.com/en/Precision_and_recall
            selected = set(predicted_bcs)
            relevant = set(correct_bcs)
            true_positives = selected & relevant
            false_positives = selected - relevant
            false_negatives = relevant - selected

            selected_count += len(selected)
            relevant_count += len(relevant)
            true_positives_count += len(true_positives)
            false_positives_count += len(false_positives)
            false_negatives_count += len(false_negatives)

    precision = true_positives_count / selected_count
    recall = true_positives_count / relevant_count

    # print(
    #    f"{convid}: precision={precision}, recall={recall}, selected={predicted_bcs}, relevant={correct_bcs}, true_positives={true_positives}, false_positives={false_positives}, false_negatives={false_negatives}")
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"{convid}: predicted bcs: {selected_count}, actual_bcs={relevant_count}")
    print(f"{convid}: precision={precision:.3f}, recall={recall:.3f}, f1={f1_score:.3f}")


def write_wavs(reader: DBReader, convs: List[str], count_per_set: int, net_version: str, bc_sample_tracks):
    random.shuffle(convs)
    for conv in convs[0:count_per_set]:
        channel = random.choice(["A", "B"])
        convchannel = f"{conv}-{channel}"
        smoothed = reader.features.get_net_output(convchannel, "best", smooth=True)
        bcs = []
        bc_sampletrack = None
        while len(bcs) < 5:
            bc_sampletrack = random.choice(bc_sample_tracks)
            bcs = list(get_bc_samples(reader, bc_sampletrack))
        bc_audio = get_bc_audio(smoothed, reader, bcs)
        print("evaluating conv {} with bc samples from {}".format(convchannel, bc_sampletrack))
        _orig_audio = reader.features.get_adc(convchannel)
        out_dir = os.path.join("evaluate", "out", net_version)
        os.makedirs(out_dir, exist_ok=True)
        minlen = min(_orig_audio.size, bc_audio.size)
        orig_audio = normalize_audio(_orig_audio[0:minlen].reshape((minlen, 1)))
        bc_audio = normalize_audio(bc_audio[0:minlen].reshape((minlen, 1)))
        audio = np.append(orig_audio, bc_audio, axis=1)
        # audio_cut = NumFeature(np.array([], dtype="float32").reshape((0, 2)))
        for start, end in get_talking_segments(reader, convchannel):
            end += 1
            length = end - start
            if length < 15:
                continue
            start_inx = reader.features.time_to_sample_index(_orig_audio, start)
            end_inx = reader.features.time_to_sample_index(_orig_audio, end)
            soundfile.write(os.path.join(out_dir, "SPK={}{} @{:.1f}s BC={}.wav".format(conv, channel, start,
                                                                                       bc_sampletrack.replace("-",
                                                                                                              ""))),
                            audio[start_inx:end_inx], 8000)
            # audio_cut = np.append(audio_cut, audio[start_inx:end_inx])

            # soundfile.write(os.path.join(out_dir, "{}--{}.wav".format(convchannel, bc_sampletrack)), audio_cut, 8000)


def output_bc_samples(reader: DBReader, convs: List[str]):
    for conv in convs:
        adc = reader.features.get_adc(conv)
        bcs = reader.get_backchannels(list(reader.get_utterances(conv)))

        def generator():
            for (bc, bcInfo) in bcs:
                from_index = reader.features.time_to_sample_index(adc, float(bcInfo['from']))
                to_index = reader.features.time_to_sample_index(adc, float(bcInfo['to']))
                yield adc[from_index:to_index]

        audio_cut = np.concatenate(generator())
        out_dir = os.path.join("tmp")
        os.makedirs(out_dir, exist_ok=True)
        soundfile.write(os.path.join(out_dir, "{}.wav".format(conv)), audio_cut, 8000)


def main():
    config_path = sys.argv[1]
    _, _, version, _ = config_path.split("/")

    config = load_config(config_path)

    conversations = read_conversations(config)
    allconversations = [convid for convlist in conversations.values() for conv in convlist for convid in
                        [conv + "-" + channel for channel in ["A", "B"]]]

    reader = DBReader(config, config_path)
    evaluate_convs(reader, sorted(conversations['eval']))
    # output_convs(conversations['eval'])
    # output_bc_samples(allconversations)
    # print("\n".join(allconversations))


if __name__ == "__main__":
    main()
