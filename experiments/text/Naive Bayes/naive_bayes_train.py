import nltk
import pickle

TRAIN_FILE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//train.txt"
ALL_WORDS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//Naive Bayes//all_words.txt"
PICKLE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//Naive Bayes//naive_bayes.pickle"

label_to_id = {
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

def get_train_data(lines, all_words):
    train_data = []
    for line in lines:
        label = line.split('\t')[1]
        utterance = line.split('\t')[2]
        if label in label_to_id:
            tokens = nltk.word_tokenize(utterance.lower())
            train_instance = ({word: (word in tokens) for word in all_words}, label_to_id[label])
            train_data.append(train_instance)
    return train_data

train_file = open(TRAIN_FILE, "r")
train_lines = train_file.readlines()

all_words = get_all_words(train_lines)

with open(ALL_WORDS, "w") as f:
    for word in all_words:
        f.write(word + '\n')

train_data = get_train_data(train_lines, all_words)

classifier = nltk.NaiveBayesClassifier.train(train_data)
classifier.show_most_informative_features()

f = open(PICKLE, 'wb')
pickle.dump(classifier, f)
f.close()