import nltk
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
import pickle

TEST_FILE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//dev.txt"
ALL_WORDS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//Naive Bayes//all_words.txt"
PICKLE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//Naive Bayes//naive_bayes.pickle"

def get_all_words(lines):
    all_words = set([])
    for line in lines:
        all_words.add(line[:-1])
    return all_words

def get_test_data(lines, all_words):
    test_data = []
    test_labels = []
    for line in lines:
        label = line.split('\t')[1]
        utterance = line.split('\t')[2]
        tokens = nltk.word_tokenize(utterance.lower())
        train_instance = {word: (word in tokens) for word in all_words}
        test_data.append(train_instance)
        test_labels.append(label)
    return test_data, test_labels

words_file = open(ALL_WORDS, "r")
word_lines = words_file.readlines()
all_words = get_all_words(word_lines)

test_file = open(TEST_FILE, "r")
test_lines = test_file.readlines()
test_data, test_labels = get_test_data(test_lines, all_words)

f = open(PICKLE, 'rb')
classifier = pickle.load(f)
f.close()

pred = classifier.classify_many(test_data)
print(pred)
print(test_labels)
print("#########################\n")
print(recall_score(test_labels, pred, average='macro'))

print("#########################\n")
cm = ConfusionMatrix(test_labels, pred)
print(cm)

print("Accuracy:" + str(accuracy(test_labels, pred)))