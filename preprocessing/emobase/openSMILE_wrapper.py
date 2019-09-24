import os
from global_vars import *
import tempfile
import numpy as np
import subprocess

OPENSMILE_PATH = os.path.join(ROOT_FOLDER, "opensmile-2.3.0//bin//Win32//SMILExtract_Release.exe")
CONFIG_PATH = os.path.join(ROOT_FOLDER, "opensmile-2.3.0//config//emobase.conf")
WAV_FOLDER = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs_word_level")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//audio//emobase_word_level")

cmdline_base = OPENSMILE_PATH + " -configfile " + CONFIG_PATH

def get_emobase_features(audio_file):
    """
    Reads the emobase feature set out of the openSMILE tool and returns the featureset as float array
    (989 features per audio file)
    :param audio_file:
    :return:
    """
    cmdline = cmdline_base + " -inputfile " + audio_file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, 'out.arff')
        cmdline += " -arffout " + temp_file
        with open(os.devnull, 'wb') as devnull:
            subprocess.call(cmdline, stdout=devnull, stderr=subprocess.STDOUT)

        with open(temp_file, "rb") as f:
            # only read last line (parts copied from https://stackoverflow.com/questions/3346430)
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            last = str(f.readline())
            # remove everything before and after the last comma to only get the features
            if len(last.split(',')) <= 1:
                print('No features for ' + audio_file)
                return
            cleaned_last_line = last.split(',', 1)[1]
            cleaned_last_line = cleaned_last_line.rsplit(',', 1)[0]
            emobase_features = np.array(cleaned_last_line.split(',')).astype(np.float)
    return emobase_features