from os import listdir
import re
from os.path import isfile, join

SKIPPED_LABELS = ["xxx", "sur", "fea", "oth", "dis"]
REMOVE_NONVERBAL = True
DIALOGUE_PATH = "C://Users//Henry//Desktop//IEMOCAP_txt//dialogue"
EMOTION_PATH = "C://Users//Henry//Desktop//IEMOCAP_txt//emotion"
EXTRACTED_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//extracted_data_skip_labels_remove_nonverbal.txt"

# clear results file
open(EXTRACTED_FILE, 'w').close()

with open(EXTRACTED_FILE, "a") as f:
    f.write("% ID; Label; Transcription\n")

def extract_labels_for_file(file_name):
    print("Extracting file " + file_name)
    kvp = {}
    with open(join(DIALOGUE_PATH, file_name)) as file:
        line = file.readline()
        while line:
            if not (line.startswith('M:') or line.startswith('F:')):
                utterance_name = line.split(" [")[0]
                utterance_transcription = line.split("]: ")[1]
                # normalize text
                utterance_transcription = utterance_transcription.replace('\t', ' ')
                utterance_transcription = utterance_transcription.replace('\n', '')

                if REMOVE_NONVERBAL:
                    utterance_transcription = re.sub(r'\[.+\]', '', utterance_transcription)

                utterance_transcription = " ".join(utterance_transcription.split())
                kvp[utterance_name] = utterance_transcription
            line = file.readline()[:-1]

    with open(join(EMOTION_PATH, file_name)) as file:
        line = file.readline()
        while line:
            if line.startswith('['):
                splits = line.split('\t')
                utterance_name = splits[1]
                utterance_emotion = splits[2]
                kvp[utterance_name] = [kvp[utterance_name], utterance_emotion]
            line = file.readline()

    with open(EXTRACTED_FILE, "a") as f:
        for key, value in kvp.items():
            if len(value) == 2 and len(value[0]) != 0:
                #  ID   Label   Utterance
                if value[1] not in SKIPPED_LABELS:
                    f.write(key + '\t' + value[1] + '\t' + value[0] + '\n')

data_dict = {}

only_files = [f for f in listdir(DIALOGUE_PATH) if isfile(join(DIALOGUE_PATH, f))]

for file in only_files:
    extract_labels_for_file(file)