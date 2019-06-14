import os
from classification.util.global_vars import *

wav_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs")
train_txt = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//text//utterances//train.txt")
test_txt = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//text//utterances//test.txt")
dev_txt = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//text//utterances//dev.txt")
out_csv = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels.tsv")

audio_name_to_path = {}
for r, d, f in os.walk(wav_folder):
    for file in f:
        if '.wav' in file:
            base_path = os.path.basename(os.path.normpath(r))
            audio_name_to_path[file[:-4]] = os.path.join(base_path, file)

def append_to_out_str(file, set, start_id, out_str):
    file_in = open(file, "r")
    file_in_lines = file_in.readlines()

    for line in file_in_lines:
        split = line[:-1].split("\t")
        name = split[0]
        label = split[1]
        set = set
        audio = audio_name_to_path[name]
        transcription = split[2]
        out_str += str(start_id) + "\t" + name + "\t" + label + "\t" + set + "\t" + audio + "\t" + transcription + "\n"
        start_id += 1
    return out_str, start_id


out_full = "Id\tName\tLabel\tSet\tAudio\tTranscription\n"
id = 0

out_full, id = append_to_out_str(train_txt, "train", id, out_full)
out_full, id = append_to_out_str(dev_txt, "dev", id, out_full)
out_full, id = append_to_out_str(test_txt, "test", id, out_full)

with open(out_csv, "w") as f:
    f.write(out_full)
