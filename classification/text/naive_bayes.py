import nltk
import pickle
from classification.util.experiments_util import log_metrics
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
from collections import OrderedDict
import os

def __get_all_words(lines):
    all_words = set([])
    for line in lines:
        utterance = line.split('\t')[2]
        tokens = nltk.word_tokenize(utterance.lower())
        all_words.update(tokens)
    return all_words

def __get_train_data(lines, all_words, label_to_id):
    train_data = []
    for line in lines:
        label = line.split('\t')[1]
        utterance = line.split('\t')[2]
        if label in label_to_id:
            tokens = nltk.word_tokenize(utterance.lower())
            train_instance = ({word: (word in tokens) for word in all_words}, label_to_id[label])
            train_data.append(train_instance)
    return train_data

def __get_test_data(lines, all_words, label_to_id):
    test_data = []
    test_labels = []
    for line in lines:
        label = line.split('\t')[1]
        utterance = line.split('\t')[2]

        if label in label_to_id:
            tokens = nltk.word_tokenize(utterance.lower())
            train_instance = {word: (word in tokens) for word in all_words}
            test_data.append(train_instance)
            test_labels.append(label_to_id[label])
    return test_data, test_labels

def __get_all_words_from_file(all_words_file):
    words_file = open(all_words_file, "r")
    word_lines = words_file.readlines()
    all_words = set([])
    for line in word_lines:
        all_words.add(line[:-1])
    return all_words

def train(train_file_in, experiment_dir, label_to_id, logger):
    logger.info("############# Training naive bayes classifier on " + str(train_file_in))

    train_file = open(train_file_in, "r")
    train_lines = train_file.readlines()

    all_words = __get_all_words(train_lines)

    all_words_file = os.path.join(experiment_dir, "allwords.txt")
    with open(all_words_file, "w") as f:
        for word in all_words:
            f.write(word + '\n')

    train_data = __get_train_data(train_lines, all_words, label_to_id)

    classifier = nltk.NaiveBayesClassifier.train(train_data)
    classifier.show_most_informative_features()

    model_path = os.path.join(experiment_dir, "naive_bayes_text.pkl")
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    logger.info("Completed training. Saved model to " + model_path)

def test(test_file_in, experiment_dir, label_to_id, logger):
    logger.info("############# Testing naive bayes classifier on " + str(test_file_in))

    all_words_file = os.path.join(experiment_dir, "allwords.txt")
    all_words = __get_all_words_from_file(all_words_file)
    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, test_labels = __get_test_data(test_lines, all_words, label_to_id)

    model_path = os.path.join(experiment_dir, "naive_bayes_text.pkl")
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()

    logger.info("Loaded model from " + str(model_path))

    pred = classifier.classify_many(test_data)

    log_metrics(test_labels, pred, logger)

def eval_get_probability_scores(test_file_in, experiment_dir, label_to_id, logger):
    logger.info("Getting naive bayes probability scores for " + str(test_file_in))

    all_words_file = os.path.join(experiment_dir, "allwords.txt")
    all_words = __get_all_words_from_file(all_words_file)

    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, _ = __get_test_data(test_lines, all_words, label_to_id)

    model_path = os.path.join(experiment_dir, "naive_bayes_text.pkl")
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()
    logger.info("Loaded model from " + str(model_path))

    sorted_labels = list(OrderedDict.fromkeys(label_to_id.values()))

    probabilities = []
    for test_vector in test_data:
        label_probs = []
        prob_dist = classifier.prob_classify(test_vector)
        for label in sorted_labels:
            label_probs.append(prob_dist.prob(label))
        probabilities.append(label_probs)

    return probabilities
