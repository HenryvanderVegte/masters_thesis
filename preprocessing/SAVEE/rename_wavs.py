import os
from global_vars import *
from shutil import copyfile

wav_folder = os.path.join(ROOT_FOLDER, "datasets//SAVEE//wavs")
out_csv = os.path.join(ROOT_FOLDER, "datasets//SAVEE//labels.tsv")

audio_path_to_name = {}
for r, d, f in os.walk(wav_folder):
    for file in f:
        if '.wav' in file:
            base_path = os.path.basename(os.path.normpath(r))
            old_path = os.path.join(r, file)
            new_path = os.path.join(wav_folder, base_path + '_' + file)
            copyfile(old_path, new_path)

