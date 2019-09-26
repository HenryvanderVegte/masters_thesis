from global_vars import *
from utils.experiments_util import *
from random import shuffle

tsv_data = read_tsv_metadata(os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels.tsv"))
tsv_data_new = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels_new.tsv")

def add_word_count(tsv_data):
    '''
    Adds number of words in utterance to tsv label file
    :param tsv_data:
    :return:
    '''
    utterances = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment//utterances")
    for instance in tsv_data:
        name = instance['Name']
        word_file = os.path.join(utterances, name + '.txt')

        if os.path.isfile(word_file):
            words = len(open(word_file).read().splitlines())
        else:
            print(name)
            words = 0

        instance['Word_Count'] = str(words)
    return tsv_data

def add_all_labels (tsv_data):
    '''
    Adds new field to IEMOCAP tsv with all annotated labels,
    for instance:
    Name                    Labels_All
    Ses01F_impro02_F000     sad;sad;fea;neu

    :param tsv_data:
    :return:
    '''
    label_map = {
        'Neutral': 'neu',
        'Surprise': 'sur',
        'Frustration': 'fru',
        'Anger': 'ang',
        'Disgust': 'dis',
        'Surprise': 'sur',
        'Excited': 'exc',
        'Sadness': 'sad',
        'Fear': 'fea',
        'Happiness': 'hap',
        'Other': 'oth'
    }

    full_labels_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels_full")
    name_to_labels = {}
    for r, d, f in os.walk(full_labels_folder):
        for file in f:
            file_path = os.path.join(r, file)
            with open(file_path, "r") as txt:
                line = txt.readline()

                while line:
                    if line.startswith("["):
                        instance = {}
                        labels = []
                        name = line.split('\t')[1]
                        instance["majority_label"] = line.split('\t')[2]

                        line = txt.readline()

                        while line.startswith('C-'):
                            if not line.startswith('C-F') and not line.startswith('C-M'):
                                labels_by_person = line.split('\t')[1]

                                for label_by_person in labels_by_person.split(' '):
                                    labels.append(label_map[label_by_person[:-1]])

                            line = txt.readline()
                        instance['labels'] = ';'.join(labels)
                        name_to_labels[name] = instance

                    line = txt.readline()

    for instance in tsv_data:
        name = instance['Name']
        instance['Labels_All'] = name_to_labels[name]['labels']
    return tsv_data

def add_gender(tsv_data):
    '''
    Add gender to tsv - gender is derived from the utterance name
    :param tsv_data:
    :return:
    '''
    for instance in tsv_data:
        name = instance['Name']
        gender = name.split('_')[-1][:1]
        instance['Gender'] = gender
    return tsv_data

def add_folds(tsv_data, nr_of_folds):
    """
    Adds each utterance to a fold, makes sure that no utterance of the same session
    are added to the same fold
    :param tsv_data:
    :param nr_of_folds: max. number of folds
    :return:
    """
    sessions = []
    for instance in tsv_data:
        # get session id from name:
        # if name is 'Ses01F_impro05_M004', the session id would be 'Ses01F_impro05'
        session = instance['Name'].split('\t')[1][:-5]
        if session not in sessions:
            sessions.append(session)

    shuffle(sessions)

    session_to_fold = {}
    for i, session in enumerate(sessions):
        session_to_fold[session] = i % nr_of_folds

    for instance in tsv_data:
        session = instance['Name'].split('\t')[1][:-5]
        fold = str(session_to_fold[session])
        instance['Fold'] = fold
    return tsv_data

def tsv_data_to_string(tsv_data):
    '''
    Creates a string based on the loaded label file.
    Loaded label file consists of a list of dictionaries, one for each data instance
    :param tsv_data:
    :return:
    '''
    keys = tsv_data[0].keys()
    tsv_str = '\t'.join(keys) + '\n'
    for instance in tsv_data:
        items = []
        for key in keys:
            items.append(instance[key])
        tsv_str += '\t'.join(items) + '\n'
    return tsv_str