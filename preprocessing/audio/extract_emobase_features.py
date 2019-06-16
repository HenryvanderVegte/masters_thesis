import os
from classification.util.global_vars import *

OPENSMILE_PATH = os.path.join(ROOT_FOLDER, "opensmile-2.3.0//bin//Win32//SMILExtract_Release.exe")
CONFIG_PATH = os.path.join(ROOT_FOLDER, "opensmile-2.3.0//config//emobase.conf")
WAV_FOLDER = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//wavs_word_level")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//audio//emobase_word_level")

audio_path_to_name = {}
for r, d, f in os.walk(WAV_FOLDER):
    for file in f:
        if '.wav' in file:
            base_path = os.path.basename(os.path.normpath(r))
            audio_path_to_name[os.path.join(base_path, file)] = (base_path, file[:-4])


cmdline_base = OPENSMILE_PATH + " -configfile " + CONFIG_PATH

for path in audio_path_to_name:
    cmdline = cmdline_base + " -inputfile " + os.path.join(WAV_FOLDER, path)

    base_dir, file_name = audio_path_to_name[path]

    base_dir = os.path.join(OUTPUT_FOLDER, base_dir)

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    output_file = os.path.join(base_dir, file_name)
    output_file_arff = output_file + ".arff"
    cmdline += " -arffout " + output_file_arff
    os.system(cmdline)

    with open(output_file_arff, "rb") as f:
        # only read last line (https://stackoverflow.com/questions/3346430)
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)
        last = str(f.readline())
        #remove everything before and after the last comma to only get the features
        cleaned_last_line = last.split(',', 1)[1]
        cleaned_last_line = cleaned_last_line.rsplit(',', 1)[0]

        cleaned_last_line = cleaned_last_line.replace(',','\n')

        output_file_emobase = output_file + ".emobase"
        with open(output_file_emobase, "w") as f:
            f.write(cleaned_last_line)
    os.remove(output_file_arff)