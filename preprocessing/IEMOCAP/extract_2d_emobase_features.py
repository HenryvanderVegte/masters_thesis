from classification.util.global_vars import *
from classification.text.word_embeddings import *
import os

'''
This script uses the recognized words, takes the emobase features on a word level and creates a dataset out of it.
'''

utterances_with_words = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//forced_alignment//utterances')
emobase_word_level = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level')
embeddings_out = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_dataset//dataset.npy')

def load_emobase_features(path):
    fl = []
    with open(path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            fl.append(float(line))

    return fl

instance_dict = {}
for r, d, f in os.walk(utterances_with_words):
    for file in f:
        file_path = os.path.join(r, file)
        name = file[:-4]
        with open(file_path, "r") as f:
            print(name)
            f_lines = f.readlines()

            instance_matrix = np.zeros(shape=(len(f_lines), 989))

            i = 0
            for line in f_lines:
                local_path = line.split('\t')[0][:-4] + '.emobase'

                emobase_features = load_emobase_features(os.path.join(emobase_word_level, local_path))
                instance_matrix[i, :] = np.array(emobase_features)
                i += 1
        instance_dict[name] = instance_matrix

np.save(embeddings_out, instance_dict)