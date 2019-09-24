import os
from global_vars import *

wav_folder = os.path.join(ROOT_FOLDER, "datasets//RAVDESS//wavs")
out_csv = os.path.join(ROOT_FOLDER, "datasets//RAVDESS//labels.tsv")

audio_path_to_name = {}
for r, d, f in os.walk(wav_folder):
    for file in f:
        if '.wav' in file:
            base_path = os.path.basename(os.path.normpath(r))
            audio_path_to_name[os.path.join(base_path, file)] = file[:-4]


def filename_to_label(filename):
    if filename.startswith('01', 6, 8):
        return 'neu'
    elif filename.startswith('02', 6, 8):
        return 'cal'
    elif filename.startswith('03', 6, 8):
        return 'hap'
    elif filename.startswith('04', 6, 8):
        return 'sad'
    elif filename.startswith('05', 6, 8):
        return 'ang'
    elif filename.startswith('06', 6, 8):
        return 'fea'
    elif filename.startswith('07', 6, 8):
        return 'dis'
    elif filename.startswith('08', 6, 8):
        return 'sur'
    else:
        print("not found" + filename)

out_full = "Id\tName\tLabel\tSet\tAudio\n"
id = 0

for path in audio_path_to_name.keys():

    label = filename_to_label(audio_path_to_name[path])
    out_full += str(id) + "\t" + audio_path_to_name[path] + "\t" + label + "\t" + "train" + "\t" + path + "\n"
    id += 1

with open(out_csv, "w") as f:
    f.write(out_full)
