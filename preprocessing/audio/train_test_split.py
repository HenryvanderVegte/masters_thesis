import random

FEATURE_FILE = "C://Users//Henry//Desktop//IEMOCAP_audio//emobase_features.txt"
TRAIN_FILE_TEXT = "C://Users//Henry//Desktop//IEMOCAP_txt//split//train.txt"
DEV_FILE_TEXT = "C://Users//Henry//Desktop//IEMOCAP_txt//split//dev.txt"
TEST_FILE_TEXT = "C://Users//Henry//Desktop//IEMOCAP_txt//split//test.txt"

TRAIN_FILE_AUDIO = "C://Users//Henry//Desktop//IEMOCAP_audio//split//train.txt"
DEV_FILE_AUDIO = "C://Users//Henry//Desktop//IEMOCAP_audio//split//dev.txt"
TEST_FILE_AUDIO = "C://Users//Henry//Desktop//IEMOCAP_audio//split//test.txt"

# clear train dev test files
open(TRAIN_FILE_AUDIO, 'w').close()
open(DEV_FILE_AUDIO, 'w').close()
open(TEST_FILE_AUDIO, 'w').close()

def get_train_test_dev_dicts():
    train_file_text = open(TRAIN_FILE_TEXT, "r")
    train_lines = train_file_text.readlines()
    train_dict = create_dict(train_lines)

    dev_file_text = open(DEV_FILE_TEXT, "r")
    dev_lines = dev_file_text.readlines()
    dev_dict = create_dict(dev_lines)

    test_file_text = open(TEST_FILE_TEXT, "r")
    test_lines = test_file_text.readlines()
    test_dict = create_dict(test_lines)

    return train_dict, dev_dict ,test_dict

def create_dict(lines):
    id_label_dict = {}
    for line in lines:
        split = line.split('\t')
        id = split[0]
        label = split[1]
        id_label_dict[id] = label
    return id_label_dict

train_dict, dev_dict, test_dict = get_train_test_dev_dicts()

feature_file = open(FEATURE_FILE, "r")
feature_file_lines = feature_file.readlines()

train_lines = []
dev_lines = []
test_lines = []
for line in feature_file_lines:
    id = line.split('\t')[0]
    values = line.split('\t')[1]
    if id in train_dict:
        train_lines.append(id + '\t' + train_dict[id] + '\t' + values)
    elif id in dev_dict:
        dev_lines.append(id + '\t' + dev_dict[id] + '\t' + values)
    elif id in test_dict:
        test_lines.append(id + '\t' + test_dict[id] + '\t' + values)

with open(TRAIN_FILE_AUDIO, "w") as f:
    f.writelines(train_lines)
with open(DEV_FILE_AUDIO, "w") as f:
    f.writelines(dev_lines)
with open(TEST_FILE_AUDIO, "w") as f:
    f.writelines(test_lines)