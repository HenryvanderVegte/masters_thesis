import nltk
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
import pickle

EXAMPLE_TEST_FILE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//dev.txt"
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
    words_file = open(all_words_file_in, "r")
    word_lines = words_file.readlines()
    all_words = get_all_words(word_lines)

    test_file = open(test_file_in, "r")
    test_lines = test_file.readlines()
    test_data, test_labels = get_test_data(test_lines, all_words, label_to_id)

    f = open(pickle_in, 'rb')
    classifier = pickle.load(f)
    f.close()

    tuples = []

    for i in range(0,len(test_data)):
        tuples.append((test_data[i], test_labels[i]))

    pred_perc = classifier.prob_classify(test_data[0])

    for i in pred_perc.samples():
        print(i + " " + str(pred_perc.prob(i)))

    print(pred_perc.prob('0'))
    print(pred_perc.prob('1'))
    print(pred_perc.prob('2'))
    print(pred_perc.prob('3'))
    print(pred_perc.prob('4'))
    print(pred_perc.prob('5'))

    pred = classifier.prob_classify_many(test_data)
    print(pred)
    print(test_labels)
    print("#########################\n")
    print(recall_score(test_labels, pred, average='macro'))

    print("#########################\n")
    cm = ConfusionMatrix(test_labels, pred)
    print(cm)

    print("Accuracy:" + str(accuracy(test_labels, pred)))

test(EXAMPLE_TEST_FILE, EXAMPLE_ALL_WORDS, EXAMPLE_PICKLE, example_label_to_id)