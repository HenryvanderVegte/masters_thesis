import nltk
import numpy as np
import gensim

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def extract_word_embeddings(file_in, model, label_file_out, vector_file_out, max_token_length):
    vector_size = model.vector_size
    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    features_per_instance = vector_size * max_token_length
    feature_vectors = np.zeros(shape=(len(file_in_lines), features_per_instance))

    label_file = ""

    for i in range(len(file_in_lines)):
        print(str(i) + " of " + str(len(file_in_lines)))
        split = file_in_lines[i].split("\t")
        id = split[0]
        label = split[1]

        label_file += id + "\t" + label + "\n"
        utterance = file_in_lines[i].split('\t')[2]
        tokens = tokenizer.tokenize(utterance.lower())

        vector = np.zeros(shape=(vector_size * max_token_length))
        j = 0
        for token in tokens[:max_token_length]:
            if token in model.wv:
                vector[j*vector_size:(j+1)*vector_size] =  np.array(model.wv[token])
            j += 1

        feature_vectors[i,:] = vector

    with open(label_file_out, "w") as f:
        f.write(label_file)

    np.save(vector_file_out, feature_vectors)

'''
extracts the feature vectors into a matrix with 
(instance, token, vector)
'''
def extract_word_embeddings_with_time(file_in, model, label_file_out, dict_file_out, oov_vector):
    vector_size = model.vector_size

    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    label_file = ""
    instance_dict = {}

    for i in range(len(file_in_lines)):
        print(str(i) + " of " + str(len(file_in_lines)))
        split = file_in_lines[i].split("\t")
        id = split[0]
        label = split[1]

        label_file += id + "\t" + label + "\n"
        utterance = file_in_lines[i].split('\t')[2]
        tokens = tokenizer.tokenize(utterance.lower())

        if len(tokens) == 0:
            continue

        instance_matrix = np.zeros(shape=(len(tokens), vector_size))
        j = 0
        for token in tokens:
            instance_matrix[j,:] = np.array(model.wv[token]) if token in model.wv else oov_vector
            j += 1

        instance_dict[id] = instance_matrix

    with open(label_file_out, "w") as f:
        f.write(label_file)

    np.save(dict_file_out, instance_dict)


