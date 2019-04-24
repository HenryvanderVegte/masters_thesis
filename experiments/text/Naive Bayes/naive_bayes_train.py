import nltk
import pickle

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

train(EXAMPLE_TRAIN_FILE, EXAMPLE_ALL_WORDS, EXAMPLE_PICKLE, example_label_to_id)