import os
from classification.util.global_vars import *
import re
from pydub import AudioSegment
from pydub.utils import make_chunks

fa_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment//wdseg")
fa_utterances = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment//utterances")
wav_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs")
wav_word_level_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs_word_level")

chunk_length_ms = 10

def read_wdseg_file(wdseg_file):
    all_words = []
    with open(wdseg_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            line = line.strip()
            line = re.sub(' +', ' ', line)
            split = line.split(' ')
            word = split[3]
            word = re.sub("[\(\[\<].*?[\)\]\>]", "", word)
            if word != '' and '++' not in word:
                all_words.append((split[0],split[1],word))
    return all_words

for r, d, f in os.walk(fa_folder):
    for file in f:
        if '.wdseg' in file:
            wdseg_file = os.path.join(r, file)
            wav_file = file[:-6] + '.wav'
            wav_file = os.path.join(wav_folder, wav_file)
            all_words = read_wdseg_file(wdseg_file)

            wav_audio = AudioSegment.from_file(wav_file, "wav")
            chunks = make_chunks(wav_audio, chunk_length_ms)

            utterance_file_text = ''

            word_wav_dir = os.path.join(wav_word_level_folder, file[:-6])
            os.mkdir(word_wav_dir)

            i = 0
            for (start, end, word) in all_words:
                word_wav_file = file[:-6] + '_' + str(i) + '_' + word + '.wav'
                utterance_file_text += file[:-6] + '//' +word_wav_file + '\t' + word + '\n'

                word_wav_file = os.path.join(word_wav_dir, word_wav_file)
                segments = chunks[int(start) - 2:int(end) +2]

                combined = AudioSegment.empty()
                for segment in segments:
                    combined += segment

                combined.export(word_wav_file, format="wav")
                i += 1

            utterance_file = os.path.join(fa_utterances, file[:-6] + '.txt')
            with open(utterance_file, 'w') as f:
                f.write(utterance_file_text)