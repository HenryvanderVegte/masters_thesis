import os
from global_vars import *

'''
Two .wav files seem to have no forced alignment - this script will identify the missing ones

-> the missing ones are 
Ses03F_impro07_M030
Ses03M_impro03_M001
'''

forced_alignment_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment")
wav_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs")

fa_names = []
for r, d, f in os.walk(forced_alignment_folder):
    for file in f:
        if '.wdseg' in file:
            fa_name = file[:-6]
            fa_names.append(fa_name)

for r, d, f in os.walk(wav_folder):
    for file in f:
        if '.wav' in file:
            wav_name = file[:-4]
            if wav_name not in fa_names:
                print(wav_name)
