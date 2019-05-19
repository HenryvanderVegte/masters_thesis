import numpy as np
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from string import punctuation
tokenizer = RegexpTokenizer(r'\w+')

def extract_tokens_from_utterance(utterance):
    utterance = ''.join([c for c in utterance if c not in punctuation])
    tokens = tokenizer.tokenize(utterance.lower())
    return tokens

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
extracts the feature vectors into a dictionary with 
dict[id] = (tokens, vectors)
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

def extract_one_hot_vectors_with_time(file_in, one_hot_dict, label_file_out, dict_file_out):
    """
    Extracts the feature vectors into a dictionary with dict[id] = (tokens, vectors).
    Vectors are one-hot-encoded based on a dictionary.
    :param file_in:
    :param one_hot_dict:
    :param label_file_out:
    :param dict_file_out:
    :return:
    """

    one_hot_dim = len(one_hot_dict["OOV"])

    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    label_file = ""
    instance_dict = {}

    for i in range(len(file_in_lines)):
        split = file_in_lines[i].split("\t")

        id = split[0]
        label = split[1]
        label_file += id + "\t" + label + "\n"

        utterance = file_in_lines[i].split('\t')[2][:-1]
        tokens = extract_tokens_from_utterance(utterance)

        if len(tokens) == 0:
            continue

        instance_matrix = np.zeros(shape=(len(tokens), one_hot_dim))
        j = 0
        for token in tokens:
            instance_matrix[j,:] = one_hot_dict[token] if token in one_hot_dict else one_hot_dict["OOV"]
            j += 1

        instance_dict[id] = instance_matrix

    with open(label_file_out, "w") as f:
        f.write(label_file)

    np.save(dict_file_out, instance_dict)

def build_one_hot_dict(file_in, vocab_size):
    """
    Creates a dictionary containing one-hot-encoded tokens.
    Items in dictionary is vocab_size + 2 (for out of vocabulary (OOV) and end of utterance (EOU))
    :param file_in:
    :param vocab_size:
    :return:
    """
    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    all_tokens = []
    for line in file_in_lines:
        utterance = line.split("\t")[2][:-1]
        tokens = __extract_tokens_from_utterance(utterance)
        all_tokens += tokens
    counter = Counter(all_tokens)

    dict = {}
    for i, (word, _) in enumerate(counter.most_common(vocab_size)):
        one_hot = np.zeros((vocab_size + 2))
        one_hot[i] = 1
        dict[word] = one_hot

    one_hot = np.zeros((vocab_size + 2))
    one_hot[vocab_size] = 1
    dict['OOV'] = one_hot

    one_hot = np.zeros((vocab_size + 2))
    one_hot[vocab_size + 1] = 1
    dict['EOU'] = one_hot
    return dict

def extract_token_to_id_vectors_with_time(file_in, token_dict, label_file_out, dict_file_out):
    """
    Extracts the feature vectors into a dictionary with one element for each token (based on the :token_dict:)
    :param file_in:
    :param token_dict:
    :param label_file_out:
    :param dict_file_out:
    :return:
    """

    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    label_file = ""
    instance_dict = {}

    for i in range(len(file_in_lines)):
        split = file_in_lines[i].split("\t")

        id = split[0]
        label = split[1]
        label_file += id + "\t" + label + "\n"

        utterance = file_in_lines[i].split('\t')[2][:-1]
        tokens = __extract_tokens_from_utterance(utterance)

        if len(tokens) == 0:
            continue

        instance_vector = np.zeros(shape=(len(tokens)))
        j = 0
        for token in tokens:
            instance_vector[j] = token_dict[token] if token in token_dict else token_dict["OOV"]
            j += 1

        instance_dict[id] = instance_vector

    with open(label_file_out, "w") as f:
        f.write(label_file)

    np.save(dict_file_out, instance_dict)

def build_token_to_id_dict(file_in, vocab_size):
    """
    Creates a dictionary mapping the :vocab_size: most common words in the training data to ids
    :vocab_size: + 2 items are in the dictionary (for out of vocabulary (OOV) and end of utterance (EOU)
    :param file_in:
    :param vocab_size:
    :return:
    """
    file_in = open(file_in, "r")
    file_in_lines = file_in.readlines()

    all_tokens = []
    for line in file_in_lines:
        utterance = line.split("\t")[2][:-1]
        tokens = __extract_tokens_from_utterance(utterance)
        all_tokens += tokens
    counter = Counter(all_tokens)

    dict = {}
    for i, (word, _) in enumerate(counter.most_common(vocab_size)):
        dict[word] = i

    dict['OOV'] = vocab_size
    dict['EOU'] = vocab_size + 1
    return dict
