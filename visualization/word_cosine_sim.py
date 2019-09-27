import gensim
import itertools
from global_vars import *
import os
import numpy as np
from scipy import spatial

model_path = os.path.join(ROOT_FOLDER, 'pretrained_embeddings//GoogleNews-vectors-negative300.bin')

def print_cosine_similarities(word_list, model):
    for pair in itertools.combinations(word_list, r=2):
        vec1 = np.array(model.wv[pair[0]])
        vec2 = np.array(model.wv[pair[1]])
        print('Words: ' + pair[0] + '\t' + pair[1] + ' - Cosine Similarity: ' + str(1 - spatial.distance.cosine(vec1, vec2)))

model = gensim.models.KeyedVectors.load_word2vec_format('file://' + model_path, binary=True)

words = ['hate', 'hell', 'fun', 'excited']
print_cosine_similarities(words, model)