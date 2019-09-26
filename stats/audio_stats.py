import os
from pydub import AudioSegment
from global_vars import *
import datetime

def print_audio_duration_from_transcriptions_file(transcriptions_file_path):
    transcriptions = open(transcriptions_file_path).read().splitlines()
    full_duration_seconds = 0.
    missing_count = 0
    for transcription in transcriptions:
        last = transcription.split('\t')[-1].split(';')
        if len(last) != 2:
            missing_count += 1
            continue
        length = float(last[0]) + float(last[1])
        full_duration_seconds += length

    print('Duration in seconds:' + str(full_duration_seconds))
    td = str(datetime.timedelta(seconds=int(full_duration_seconds)))
    print('Duration formatted:' + td)

def print_audio_duration_from_audiofiles(folder):
    full_duration_seconds = 0.

    for r, d, f in os.walk(folder):
        for file in f:
            file_path = os.path.join(r, file)

            audio = AudioSegment.from_file(file_path)
            duration_seconds = audio.duration_seconds
            full_duration_seconds += duration_seconds

    print('Duration in seconds:' + full_duration_seconds)
    td = datetime.timedelta(seconds=int(full_duration_seconds))
    print('Duration formatted:' + td)

