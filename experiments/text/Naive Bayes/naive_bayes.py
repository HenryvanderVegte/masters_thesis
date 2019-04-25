import nltk
import pickle
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy

EXAMPLE_TRAIN_FILE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//train.txt"
EXAMPLE_ALL_WORDS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//Naive Bayes//all_words.txt"
EXAMPLE_PICKLE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//Naive Bayes//naive_bayes.pickle"

example_label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

def get_all_words(lines):
    all_words = set([])
    for line in lines:
        utterance = line.split('\t')[2]
        tokens = nltk.word_tokenize(utterance.lower())
        all_words.update(tokens)
    return all_words

def get_train_data(lines, all_words, label_to_id):
    train_data = []
    for line in lines:
        label = line.split('\t')[1]
        utterance = line.split('\t')[2]
        if label in label_to_id:
            tokens = nltk.word_tokenize(utterance.lower())
            train_instance = ({word: (word in tokens) for word in all_words}, label_to_id[label])
            train_data.append(train_instance)
    return train_data

def train(train_file_in, all_words_file_out, pickle_out, label_to_id):
    train_file = open(train_file_in, "r")
    train_lines = train_file.readlines()

    all_words = get_all_words(train_lines)

    with open(all_words_file_out, "w") as f:
        for word in all_words:
            f.write(word + '\n')

    train_data = get_train_data(train_lines, all_words, label_to_id)

    classifier = nltk.NaiveBayesClassifier.train(train_data)
    classifier.show_most_informative_features()

    f = open(pickle_out, 'wb')
    pickle.dump(classifier, f)
    f.close()

def get_all_words_from_file(all_words_file):
    words_file = open(all_words_file, "r")
    word_lines = words_file.readlines()
    all_words = set([])
    for line in word_lines:
        all_words.add(line[:-1])
    return all_words

def get_test_data(lines, all_words, label_to_id):
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

def test(test_file_in, all_words_file_in, pickle_in, label_to_id):
    all_words = get_all_words_from_file(all_words_file_in)

    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, test_labels = get_test_data(test_lines, all_words, label_to_id)

    f = open(pickle_in, 'rb')
    classifier = pickle.load(f)
    f.close()

    pred = classifier.prob_classify_many(test_data)
    print(pred)
    print(test_labels)
    print("#########################\n")
    print(recall_score(test_labels, pred, average='macro'))

    print("#########################\n")
    cm = ConfusionMatrix(test_labels, pred)
    print(cm)

    print("Accuracy:" + str(accuracy(test_labels, pred)))

def eval_get_probability_scores(test_file_in, all_words_file_in, pickle_in, label_to_id):
    all_words = get_all_words_from_file(all_words_file_in)

    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, _ = get_test_data(test_lines, all_words, label_to_id)

    f = open(pickle_in, 'rb')
    classifier = pickle.load(f)
    f.close()

    probabilities = []
    for test_vector in test_data:
        label_probs = []
        prob_dist = classifier.prob_classify(test_vector)
        for label in label_to_id.values():
            label_probs.append(prob_dist.prob(label))
        probabilities.append(label_probs)
    return probabilities

train(EXAMPLE_TRAIN_FILE, EXAMPLE_ALL_WORDS, EXAMPLE_PICKLE, example_label_to_id)