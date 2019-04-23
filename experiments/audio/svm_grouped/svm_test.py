import pickle
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
import numpy as np

PICKLE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//svm_grouped//svm.pickle"
TEST_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//dev.txt"
MEANS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//svm_grouped//means.txt"
STDDEVS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//svm_grouped//stddevs.txt"


means = []
means_file = open(MEANS, "r")
means_file_lines = means_file.readlines()
for line in means_file_lines:
    means.append(float(line))
means = np.array(means)

stddevs = []
stddevs_file = open(STDDEVS, "r")
stddevs_file_lines = stddevs_file.readlines()
for line in stddevs_file_lines:
    stddevs.append(float(line))
stddevs = np.array(stddevs)



label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

test_vectors = []
test_labels = []

test_file = open(TEST_FILE_AUDIO, "r")
test_file_lines = test_file.readlines()
for line in test_file_lines:
    line = line.replace('\n', '')
    split = line.split('\t')
    label = split[1]
    features = split[2]
    if label in label_to_id:
        test_vectors.append(np.array(features.split(',')).astype(np.float))
        test_labels.append(int(label_to_id[label]))

normed_vectors = (np.array(test_vectors) - means) / stddevs

f = open(PICKLE, 'rb')
classifier = pickle.load(f)
f.close()

pred = np.array(classifier.predict(normed_vectors))
pred = [str(i) for i in pred]

test_labels = np.array(test_labels)
test_labels = [str(i) for i in test_labels]

print(pred)
print(test_labels)

print("#########################\n")
print(recall_score(test_labels, pred, average='macro'))

print("#########################\n")

cm = ConfusionMatrix(test_labels, pred)
print(cm)

print("Accuracy:" + str(accuracy(test_labels, pred)))