from utils.experiments_util import *
from classification.util.global_vars import *
import re
from classification.util.global_vars import *
from classification.text.word_embeddings import *
import gensim
import os
import tqdm
import numpy as np


np.random.seed(1)
oov_vector = np.random.normal(0, 0.1, 300)

model = gensim.models.KeyedVectors.load_word2vec_format('file://C://Users//Henry//Desktop//Masterarbeit//pretrained_embeddings//GoogleNews-vectors-negative300.bin', binary=True)

devtxt = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//dev.txt')
embeddings_path = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_dev.txt')

txt_lines = open(devtxt).read().splitlines()


instance_dict = {}

for line in tqdm.tqdm(txt_lines):
    instance_matrix = []

    split = line.split('\t')
    name = split[0]

    for i in range(1, len(split), 2):
        word = split[i]
        if word in model.wv:
            instance_matrix.append(np.array(model.wv[word]))
        else:
            instance_matrix.append(oov_vector)

    if not instance_matrix:
        continue

    instance_matrix = np.vstack(instance_matrix)
    instance_dict[name] = instance_matrix

np.save(embeddings_path, instance_dict)