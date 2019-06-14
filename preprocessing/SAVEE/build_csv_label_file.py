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
    if filename.startswith('a'):
        return 'ang'
    elif filename.startswith('d'):
        return 'dis'
    elif filename.startswith('f'):
        return 'fea'
    elif filename.startswith('h'):
        return 'hap'
    elif filename.startswith('n'):
        return 'neu'
    elif filename.startswith('sa'):
        return 'sad'
    elif filename.startswith('su'):
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
