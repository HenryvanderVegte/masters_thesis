import os
from shutil import copyfile
import math
import pydub

def copy_wavs(input_folder, output_folder):
    """
    Copies each .wav file of an input folder (recursive) to an output folder (flat)
    :param input_folder:
    :param output_folder:
    :return:
    """
    for r, d, f in os.walk(input_folder):
        for file in f:
            if '.wav' in file:
                print(file)
                base_path = os.path.basename(os.path.normpath(r))
                old_path = os.path.join(r, file)
                new_path = os.path.join(output_folder, file)
                copyfile(old_path, new_path)

def get_oovs_latex_format(oov_words):
    """
    Gets a list of elements and put them in rows in a format that is accepted by latex.
    Example output:
    a&doesnt&naturey&therell\\
    amys&doughnut&neednt&to\\
    .
    .
    .
    :param oov_words:
    :return:
    """
    oov_words.sort()
    elements_per_row = 4
    rows = math.ceil((len(oov_words) / elements_per_row))

    lines = ['' for i in range(rows)]

    for i in range(len(oov_words)):
        lines[i % rows] = lines[i % rows] + '&' + oov_words[i]

    for line in lines:
        line = line[1:] + '\\\\'
        print(line)

def print_unwanted_sample_rates(wav_folder):
    """
    Checks if any .wav file in folder contains an unwanted sample rate (!= 16000)
    :param wav_folder:
    :return:
    """
    for r, d, f in os.walk(wav_folder):
        for file in f:
            if '.wav' in file:
                path = os.path.join(r, file)
                audio = pydub.AudioSegment.from_wav(path)
                if str(audio.frame_rate) != "16000":
                    print(audio.frame_rate)

def rename_wavs(wav_folder):
    """
    Iterates over a folder (recursive) and copies every .wav file to the base path
    :param wav_folder:
    :return:
    """
    for r, d, f in os.walk(wav_folder):
        for file in f:
            if '.wav' in file:
                old_path = os.path.join(r, file)
                new_path = os.path.join(wav_folder, file)
                copyfile(old_path, new_path)