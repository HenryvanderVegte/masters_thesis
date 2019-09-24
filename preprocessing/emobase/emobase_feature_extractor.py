import os
from global_vars import *
import re
from pydub import AudioSegment
from pydub.utils import make_chunks
import tempfile
import preprocessing.emobase.openSMILE_wrapper as opensmile

chunk_length_ms = 10
add_extra_chunks = 4

def get_wdseg_word_level_timestamps(wdseg_file):
    """
    Returns all words with its word level timestamps based on the given utterance name:
    I.e:
    utterance_name = 'Ses01F_impro01_F000'
    words_with_timestamps =
    [
        {'word': 'EXCUSE', 'start': '45', 'end': '79'},
        {'word': 'ME', 'start': '80', 'end': '105'}
    ]
    :param utterance_name:
    :return:
    """
    words_with_timestamps = []

    if not os.path.isfile(wdseg_file):
        return

    with open(wdseg_file, "r") as f:
        lines = f.readlines()
        word_id =0
        for line in lines[1:-1]:
            line = line.strip()
            line = re.sub(' +', ' ', line)
            split = line.split(' ')
            word = split[3]
            word = re.sub("[\(\[\<].*?[\)\]\>]", "", word)
            if word != '' and '++' not in word:
                wlt = {}
                wlt['word'] = word
                wlt['start'] = int(split[0])
                wlt['end'] = int(split[1])
                wlt['id'] = word_id
                words_with_timestamps.append(wlt)
                word_id += 1
    return words_with_timestamps

def get_word_level_timestamps(audio_info):
    """
    Returns all words with its word level timestamps based on the given utterance name:
    I.e:
    utterance_name = 'Ses01F_impro01_F000'
    words_with_timestamps =
    [
        {'word': 'EXCUSE', 'start': '45', 'end': '79'},
        {'word': 'ME', 'start': '80', 'end': '105'}
    ]
    :param utterance_name:
    :return:
    """
    words_with_timestamps = []

    split = audio_info.split('\t')

    word_id = 0
    for i in range(1, len(split), 2):
        wlt = {}

        wlt['word'] = split[i]

        start = int(float(split[i+1].split(';')[0]) * 100)
        duration = int(float(split[i+1].split(';')[1]) * 100)

        wlt['start'] = start
        wlt['end'] = start + duration
        wlt['id'] = word_id
        words_with_timestamps.append(wlt)
        word_id += 1

    return words_with_timestamps

def create_word_level_wavs(wav_file, words_with_timestamps, out_folder):
    """
    Creates word level wave files based on words_with_timestamps, writes them into out_folder and
    returns their paths
    :param wav_file:
    :param words_with_timestamps:
    :param out_folder:
    :return:
    """
    word_level_wavs = []
    wav_audio = AudioSegment.from_file(wav_file, "wav")
    chunks = make_chunks(wav_audio, chunk_length_ms)
    chunk_length = len(chunks)
    wav_file_name = os.path.basename(wav_file)[:-4]

    for word_with_timestamps in words_with_timestamps:
        word_wav_file = wav_file_name + '_' + str(word_with_timestamps['id']) + '_' + word_with_timestamps['word'] + '.wav'
        word_wav_file = os.path.join(out_folder, word_wav_file)

        start = max(word_with_timestamps['start'] - add_extra_chunks, 0)
        end = min(word_with_timestamps['end'] + add_extra_chunks, chunk_length)

        segments = chunks[int(start):int(end)]
        combined = AudioSegment.empty()
        for segment in segments:
            combined += segment
        combined.export(word_wav_file, format="wav")
        word_level_wavs.append(word_wav_file)
    return word_level_wavs

def create_word_level_audios_with_extra_chunks(audio_file, words_with_timestamps, out_folder, format='wav'):
    """
    Creates word level wave files based on words_with_timestamps, writes them into out_folder and
    returns their paths
    :param audio_file:
    :param words_with_timestamps:
    :param out_folder:
    :return:
    """
    word_level_audios = []
    audio_segments = AudioSegment.from_file(audio_file, format)
    chunks = make_chunks(audio_segments, chunk_length_ms)
    chunk_length = len(chunks)
    wav_file_name = os.path.basename(audio_file)[:-4]

    for word_with_timestamps in words_with_timestamps:
        word_audio_file = wav_file_name + '_' + str(words_with_timestamps[i]['id']) + '_' + words_with_timestamps[i]['word'] + '.wav'
        word_audio_file = os.path.join(out_folder, word_audio_file)

        start = max(word_with_timestamps['start'] - add_extra_chunks, 0)
        end = min(word_with_timestamps['end'] + add_extra_chunks, chunk_length)

        segments = chunks[int(start):int(end)]
        combined = AudioSegment.empty()
        for segment in segments:
            combined += segment
        combined.export(word_audio_file, format='wav')
        word_level_audios.append(word_audio_file)
    return word_level_audios

def create_word_level_audios_with_pauses(audio_file, words_with_timestamps, out_folder, format='wav'):
    """
    Creates word level wave files based on words_with_timestamps, writes them into out_folder and
    returns their paths
    :param audio_file:
    :param words_with_timestamps:
    :param out_folder:
    :return:
    """
    word_level_audios = []
    audio_segments = AudioSegment.from_file(audio_file, format)
    chunks = make_chunks(audio_segments, chunk_length_ms)
    chunk_length = len(chunks)
    wav_file_name = os.path.basename(audio_file)[:-4]

    for i in range(len(words_with_timestamps)):
        word_audio_file = wav_file_name + '_' + str(words_with_timestamps[i]['id']) + '_' + words_with_timestamps[i]['word'] + '.wav'
        word_audio_file = os.path.join(out_folder, word_audio_file)

        if i == 0:
            start = 0
        else:
            start = words_with_timestamps[i-1]['end']

        if i == len(words_with_timestamps) - 1:
            end = chunk_length
        else:
            end = words_with_timestamps[i+1]['start']

        segments = chunks[int(start):int(end)]
        combined = AudioSegment.empty()
        for segment in segments:
            combined += segment
        combined.export(word_audio_file, format='wav')
        word_level_audios.append(word_audio_file)
    return word_level_audios

def extract_word_level_emobase_features_wdseg(utterance_name):
    wdseg_file = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment//wdseg", utterance_name + ".wdseg")
    wav_file = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs", utterance_name + ".wav")
    words_with_timestamps = get_wdseg_word_level_timestamps(wdseg_file)

    all_emobase_features = []

    if words_with_timestamps is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            word_level_wavs = create_word_level_audios_with_pauses(wav_file, words_with_timestamps, temp_dir)
            for word_level_wav in word_level_wavs:
                emobase_features = opensmile.get_emobase_features(word_level_wav)
                if emobase_features is not None:
                    all_emobase_features.append(emobase_features)

    print('completed ' + utterance_name)
    return all_emobase_features

def extract_word_level_emobase_features(audio_info):
    utterance_name = audio_info.split('\t')[0].replace('/','//')
    wav_file = os.path.join(ROOT_FOLDER, "datasets//MS//audio//dev", utterance_name)
    words_with_timestamps = get_word_level_timestamps(audio_info)

    all_emobase_features = []

    if words_with_timestamps is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            word_level_wavs = create_word_level_audios_with_pauses(wav_file, words_with_timestamps, temp_dir, format='m4a')
            for word_level_wav in word_level_wavs:
                emobase_features = opensmile.get_emobase_features(word_level_wav)
                if emobase_features is not None:
                    all_emobase_features.append(emobase_features)

    print('completed ' + utterance_name)
    return all_emobase_features

def extract_utterance_level_emobase_features(utterance_name):
    wav_file = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs", utterance_name + ".wav")
    emobase_features = opensmile.get_emobase_features(wav_file)

    print('completed ' + utterance_name)
    return emobase_features