import os
from pydub import AudioSegment
from classification.util.global_vars import *
import wave
import contextlib

wav_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs")

lengths_impro = []
lengths_script = []


for r, d, f in os.walk(wav_folder):
    for file in f:
        file_path = os.path.join(r, file)

        audio = AudioSegment.from_file(file_path)

        with contextlib.closing(wave.open(file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            if 'impro' in file:
                lengths_impro.append(duration)
            else:
                lengths_script.append(duration)

print('Impro lengths:' + str(len(lengths_impro)))
print('Script lengths:' + str(len(lengths_script)))

print('----')
for e in lengths_impro:
    print(e)
print('----')
for e in lengths_script:
    print(e)