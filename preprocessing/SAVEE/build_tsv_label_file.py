import os
from classification.util.global_vars import *

wav_folder = os.path.join(ROOT_FOLDER, "datasets//SAVEE//wavs")
out_csv = os.path.join(ROOT_FOLDER, "datasets//SAVEE//labels.tsv")

audio_path_to_name = {}
for r, d, f in os.walk(wav_folder):
    for file in f:
        if '.wav' in file:
            base_path = os.path.basename(os.path.normpath(r))
            audio_path_to_name[os.path.join(base_path, file)] = file[:-4]

def filename_to_label(filename):
    if filename.startswith('a', 3):
        return 'ang'
    elif filename.startswith('d', 3):
        return 'dis'
    elif filename.startswith('f', 3):
        return 'fea'
    elif filename.startswith('h', 3):
        return 'hap'
    elif filename.startswith('n', 3):
        return 'neu'
    elif filename.startswith('sa', 3):
        return 'sad'
    elif filename.startswith('su', 3):
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
