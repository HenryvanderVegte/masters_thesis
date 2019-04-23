import random
EXTRACTED_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//extracted_data_skip_labels_remove_nonverbal.txt"

TRAIN_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//split//train.txt"
DEV_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//split//dev.txt"
TEST_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//split//test.txt"

# clear train dev test files
open(TRAIN_FILE, 'w').close()
open(DEV_FILE, 'w').close()
open(TEST_FILE, 'w').close()


TRAIN_PERCENTAGE = 0.8
DEV_PERCENTAGE = 0.1


def get_sessions(lines):
    sessions_set = set([])
    for line in lines[1:]:
        id = line.split('\t')[0]
        id_split = id.split('_')
        session_id = id_split[0] + '_' + id_split[1]
        sessions_set.add(session_id)
    return sessions_set

def get_session_split(sessions_set):
    sessions_set_length = float(len(sessions_set))

    train_sessions_len = int(sessions_set_length * TRAIN_PERCENTAGE)
    train_sessions = set(random.sample(sessions_set, train_sessions_len))
    sessions_set -= train_sessions

    dev_sessions_len = int(sessions_set_length * DEV_PERCENTAGE)
    dev_sessions = set(random.sample(sessions_set, dev_sessions_len))
    sessions_set -= dev_sessions

    test_sessions = sessions_set
    return train_sessions, dev_sessions, test_sessions

def print_sessions_to_files(train_sessions, dev_sessions, test_sessions, lines):
    train_lines= []
    test_lines= []
    dev_lines = []
    for line in lines[1:]:
        id = line.split('\t')[0]
        id_split = id.split('_')
        session_id = id_split[0] + '_' + id_split[1]
        if session_id in train_sessions:
            train_lines.append(line)
        elif session_id in test_sessions:
            test_lines.append(line)
        elif session_id in dev_sessions:
            dev_lines.append(line)
    with open(TRAIN_FILE, "a") as f:
        f.writelines(train_lines)
    with open(DEV_FILE, "a") as f:
        f.writelines(test_lines)
    with open(TEST_FILE, "a") as f:
        f.writelines(dev_lines)

text_file = open(EXTRACTED_FILE, "r")
lines = text_file.readlines()

sessions_set = get_sessions(lines)
train, test, dev = get_session_split(sessions_set)
print_sessions_to_files(train, test, dev, lines)
