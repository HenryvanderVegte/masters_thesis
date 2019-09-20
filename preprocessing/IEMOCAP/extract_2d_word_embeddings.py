import re
from classification.util.global_vars import *
from classification.text.word_embeddings import *
import gensim
import os

'''
This script uses the recognized words with an existing word level audio and extracts pretrained word embeddings for it.
'''

utterances_with_words = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//forced_alignment//utterances')
embeddings_out = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//text//google_news_word_embeddings_1.npy')
model = gensim.models.KeyedVectors.load_word2vec_format('file://C://Users//Henry//Desktop//Masterarbeit//pretrained_embeddings//GoogleNews-vectors-negative300.bin', binary=True)

vector_size = model.vector_size

instance_dict = {}
oov_words= []
oov_vector = np.random.normal(0, 0.1, 300)
full_fl = []
for r, d, f in os.walk(utterances_with_words):
    for file in f:
        file_path = os.path.join(r, file)
        name = file[:-4]
        with open(file_path, "r") as f:
            f_lines = f.readlines()

            instance_matrix = np.zeros(shape=(len(f_lines), vector_size))

            i = 0
            for line in f_lines:
                word = line[:-1].split('\t')[1]

                word = word.lower()
                word = re.sub(r'\W+', '', word)

                if word in model.wv:
                    instance_matrix[i, :] = np.array(model.wv[word])
                else:
                    if word not in oov_words:
                        oov_words.append(word)
                    instance_matrix[i, :] = oov_vector

                full_fl.append(instance_matrix[i])
                i += 1
        instance_dict[name] = instance_matrix

np.save(embeddings_out, instance_dict)