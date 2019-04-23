import os

OPENSMILE_PATH = "C://Users//Henry//Desktop//opensmile-2.3.0//opensmile-2.3.0//bin//Win32//SMILExtract_Release.exe"
CONFIG_PATH = "C://Users//Henry//Desktop//opensmile-2.3.0//opensmile-2.3.0//config//emobase.conf"
WAV_FOLDER = "C://Users//Henry//Desktop//IEMOCAP_wav"
OUTPUT_FOLDER = "C://Users//Henry//Desktop//IEMOCAP_audio"
FEATURE_FILE = "C://Users//Henry//Desktop//IEMOCAP_audio//emobase_features.txt"

filepaths = []
filenames = []
for r, d, f in os.walk(WAV_FOLDER):
    for file in f:
        if '.wav' in file:
            filepaths.append(os.path.join(r, file))
            filenames.append(file)
print(filenames)

cmdline_base = OPENSMILE_PATH + " -configfile " + CONFIG_PATH

features = []

for i in range(len(filepaths)):
    cmdline = cmdline_base + " -inputfile " + filepaths[i]
    output_file = os.path.join(OUTPUT_FOLDER, filenames[i]) + ".arff"
    cmdline += " -arffout " + output_file
    os.system(cmdline)

    with open(output_file, "rb") as f:
        # only read last line (https://stackoverflow.com/questions/3346430)
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)
        last = str(f.readline())
        #remove everything before and after the last comma to only get the features
        cleaned_last_line = last.split(',', 1)[1]
        cleaned_last_line = cleaned_last_line.rsplit(',', 1)[0]
        features.append(filenames[i].split('.')[0] + '\t' + cleaned_last_line)

    os.remove(output_file)

with open(FEATURE_FILE, "w") as f:
    for line in features:
        f.write(line + '\n')