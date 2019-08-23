import os
from classification.util.global_vars import *
from shutil import copyfile

input_folder = 'E://masters_thesis//datasets//MSP_IMPROV//sessions'
output_folder = 'E://masters_thesis//datasets//MSP_IMPROV//wavs'

for r, d, f in os.walk(input_folder):
    for file in f:
        if '.wav' in file:
            print(file)
            base_path = os.path.basename(os.path.normpath(r))
            old_path = os.path.join(r, file)
            new_path = os.path.join(output_folder, file)
            copyfile(old_path, new_path)
