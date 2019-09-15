from utils.experiments_util import *
import nltk
import os
import pickle
import re

regex = re.compile('[^A-Za-z0-9 ]+')

def create_all_words_set_from_metadata(metadata, class_groups, folds):
    all_words = set([])
    for instance in metadata:
        if instance["Label"] not in class_groups or int(instance["Fold"]) not in folds:
            continue
        transcription = regex.sub('', instance['Transcription'].lower())
        tokens = nltk.word_tokenize(transcription)

        all_words.update(tokens)
    return all_words

def create_dataset_from_metadata(metadata, class_groups, all_words_set,  folds):
    dataset = []
    for instance in metadata:
        if instance["Label"] not in class_groups or int(instance["Fold"]) not in folds:
            continue

        label = class_groups[instance["Label"]]
        transcription = regex.sub('', instance['Transcription'].lower())
        tokens = nltk.word_tokenize(transcription)

        dataset_instance = ({word: (word in tokens) for word in all_words_set}, label)
        dataset.append(dataset_instance)
    return dataset

def train(train_dataset, test_dataset, experiment_path, logger):
    classifier = nltk.NaiveBayesClassifier.train(train_dataset)
    classifier.show_most_informative_features(n=400)

    model_path = os.path.join(experiment_path, "model.pkl")
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    test_data = [i[0] for i in test_dataset]
    test_labels = [i[1] for i in test_dataset]
    test_predictions = classifier.classify_many(test_data)

    logger.info(get_metrics_str(test_labels, test_predictions))

    return test_labels, test_predictions