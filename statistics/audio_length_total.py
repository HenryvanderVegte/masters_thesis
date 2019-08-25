import os
from pydub import AudioSegment
from tqdm import tqdm
import wave
import contextlib

input_folder = "E://masters_thesis//datasets//MS//sentiment dataset"

length = 0

paths = []

for r, d, f in tqdm(os.walk(input_folder)):
    for file in f:
        paths.append(os.path.join(r, file))

for file_path in tqdm(paths):
    if file_path.endswith('.wav'):
        with contextlib.closing(wave.open(file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            length += duration

    if file_path.endswith('.m4a'):
        audio = AudioSegment.from_file(file_path)
        length += audio.duration_seconds

print('length in seconds:' + str(length))