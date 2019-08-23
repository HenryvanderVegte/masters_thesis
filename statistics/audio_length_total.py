import os
from classification.util.global_vars import *
import wave
import contextlib

input_folder = "E://masters_thesis//datasets//MS//sentiment dataset"

length = 0


for r, d, f in os.walk(input_folder):
    for file in f:
        file_path = os.path.join(r, file)
        with contextlib.closing(wave.open(file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            length += duration

print('length:' + str(length))