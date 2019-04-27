import nltk
import numpy as np
import gensim


def extract_word_embeddings_to_file(file_in, model, file_out, max_token_length):
    vector_size = model.vector_size
    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    for line in file_in_lines:
        split = line.split("\t")
        id = split[0]
        label = split[1]

        print(id)

        utterance = line.split('\t')[2]
        tokens = nltk.word_tokenize(utterance.lower())

        vector = np.zeros(shape=(vector_size * max_token_length))
        i = 0
        for token in tokens[:max_token_length]:
            if token in model.wv:
                vector[i*vector_size:(i+1)*vector_size] =  np.array(model.wv[token])
            i += 1

        vector_as_string = ""
        for val in vector:
            vector_as_string += str(val) + ","

        with open(file_out, "a+") as f:
            f.write(id + "\t" + label + "\t" + vector_as_string[:-1] + '\n')
