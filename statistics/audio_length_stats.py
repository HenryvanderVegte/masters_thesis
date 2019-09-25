import os
from pydub import AudioSegment
from global_vars import *
import wave
import contextlib
import time

wav_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs")
wavs_folder = os.path.join(ROOT_FOLDER, "datasets//MS//audio//dev")

lengths_impro = []
lengths_script = []


def print_audio_duration(folder):
    full_duration = 0.

    i = 0
    for r, d, f in os.walk(folder):
        for file in f:
            file_path = os.path.join(r, file)

            audio = AudioSegment.from_file(file_path)
            duration_seconds = audio.duration_seconds
            full_duration += duration_seconds
            i += 1
            print(i)

    duration_str = time.strftime('%H:%M:%S', time.gmtime(int(full_duration)))
    print('Duration:' + duration_str)


print_audio_duration(wavs_folder)


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