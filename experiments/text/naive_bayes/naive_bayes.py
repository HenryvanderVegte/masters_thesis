import nltk
import pickle
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
import datetime
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

def train(train_file_in, experiment_dir, label_to_id, logfile):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log = "[" + str(timestamp) + "] Training naive bayes classifier on " + str(train_file_in) + "...\n"

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

    model_path = os.path.join(experiment_dir, "model.pkl")
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log += "[" + str(timestamp) + "] Completed training. Saved model to " + model_path + "\n"
    with open(logfile, 'a') as d:
        d.write(log)

def test(test_file_in, experiment_dir, label_to_id, logfile):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log = "[" + str(timestamp) + "] Testing naive bayes classifier on " + str(test_file_in) + "...\n"

    all_words_file = os.path.join(experiment_dir, "allwords.txt")
    all_words = __get_all_words_from_file(all_words_file)
    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, test_labels = __get_test_data(test_lines, all_words, label_to_id)

    model_path = os.path.join(experiment_dir, "model.pkl")
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()
    log += "Loaded model from " + str(model_path) + "\n"

    pred = classifier.classify_many(test_data)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log += "[" + str(timestamp) + "] Completed test. Results: \n"
    log += "Accuracy: " + str(accuracy(test_labels, pred))
    log += "Unweighted average recall: " + str(recall_score(test_labels, pred, average='macro'))
    cm = ConfusionMatrix(test_labels, pred)
    log += "Confusion Matrix:\n" + str(cm) + "\n"

    if logfile is None:
        print(log)
    else:
        with open(logfile, 'a') as d:
            d.write(log)

def eval_get_probability_scores(test_file_in, experiment_dir, label_to_id, logfile):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log = "[" + str(timestamp) + "] Getting naive bayes probability scores on " + str(test_file_in) + "...\n"

    all_words_file = os.path.join(experiment_dir, "allwords.txt")
    all_words = __get_all_words_from_file(all_words_file)

    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, _ = __get_test_data(test_lines, all_words, label_to_id)

    model_path = os.path.join(experiment_dir, "model.pkl")
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()
    log += "Loaded model from " + str(model_path) + "\n"

    probabilities = []
    for test_vector in test_data:
        label_probs = []
        prob_dist = classifier.prob_classify(test_vector)
        for label in set(list(label_to_id.values())):
            label_probs.append(prob_dist.prob(label))
        probabilities.append(label_probs)

    if logfile is None:
        print(log)
    else:
        with open(logfile, 'a') as d:
            d.write(log)

    return probabilities

#train(EXAMPLE_TRAIN_FILE, EXAMPLE_ALL_WORDS, EXAMPLE_PICKLE, example_label_to_id)