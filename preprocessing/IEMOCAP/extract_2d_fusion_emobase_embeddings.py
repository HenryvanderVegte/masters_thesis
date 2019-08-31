import re
from classification.util.global_vars import *
from classification.text.word_embeddings import *
from classification.util.experiments_util import *
import gensim
import os

'''
This script uses the recognized words with an existing word level audio and extracts pretrained words embeddings for it.
'''
metadata = read_tsv_dataset(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
utterances_with_words = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//forced_alignment//utterances')
emobase_word_level = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level')
fusion_out = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//late_fusion//emobase_and_embeddings_no_normalization.npy')
model = gensim.models.KeyedVectors.load_word2vec_format('file://C://Users//Henry.000//Desktop//Masterarbeit//pretrained_embeddings//GoogleNews-vectors-negative300.bin', binary=True)

oov_vector = np.random.normal(0, 0.1, model.vector_size)
vector_size = model.vector_size

target_labels = [] #['hap', 'exc', 'sad', 'ang', 'neu']

name_to_label = {}
for instance in metadata:
    name_to_label[instance['Name']] = instance['Label']

def load_emobase_features(path):
    fl = []
    with open(path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            fl.append(float(line))

    return fl

instance_dict = {}
full_fl = []
for r, d, f in os.walk(utterances_with_words):
    for file in f:
        file_path = os.path.join(r, file)
        name = file[:-4]

        if name not in name_to_label:
            continue

        label = name_to_label[name]

        if len(target_labels) != 0 and label not in target_labels:
            continue


        with open(file_path, "r") as f:
            f_lines = f.readlines()

            instance_matrix = np.zeros(shape=(len(f_lines), vector_size +989))

            i = 0
            for line in f_lines:
                local_path = line.split('\t')[0][:-4] + '.emobase'

                emobase_features = np.array(load_emobase_features(os.path.join(emobase_word_level, local_path)))

                word = line[:-1].split('\t')[1]
                word = word.lower()
                word = re.sub(r'\W+', '', word)
                word_vec = np.array(model.wv[word]) if word in model.wv else oov_vector

                instance_matrix[i, 0:989] = emobase_features
                instance_matrix[i, 989:vector_size + 989] = word_vec
                full_fl.append(instance_matrix[i])
                i += 1
        instance_dict[name] = instance_matrix

'''
fl = np.array(full_fl)
means = fl.mean(axis=0)
stddevs = fl.std(axis=0)
stddevs[stddevs == 0] = 1

for key in instance_dict.keys():
    instance_dict[key] = (instance_dict[key] - means) / stddevs
'''
np.save(fusion_out, instance_dict)