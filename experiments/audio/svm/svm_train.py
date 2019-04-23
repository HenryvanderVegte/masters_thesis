import numpy as np
from sklearn.svm import SVC
import pickle

TRAIN_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
PICKLE = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//svm//svm.pickle"
MEANS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//svm//means.txt"
STEDDEVS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//svm//stddevs.txt"

label_to_id = {
    "sad":"0",
    "neu":"1",
    "fru":"2",
    "hap":"3",
    "exc":"4",
    "ang":"5",
}

train_file = open(TRAIN_FILE_AUDIO, "r")
train_file_lines = train_file.readlines()

train_vectors = []
train_labels = []

for line in train_file_lines:
    line = line.replace('\n', '')
    split = line.split('\t')
    label = split[1]
    features = split[2]

    train_labels.append(label_to_id[label])
    train_vectors.append(np.array(features.split(',')).astype(np.float))

train_vectors = np.array(train_vectors)

means = train_vectors.mean(axis=0)
stddevs = train_vectors.std(axis=0)

#remove 0 values
stddevs[stddevs==0] = 1

normed_vectors = (train_vectors - means) / stddevs

classifier = SVC(gamma='scale', decision_function_shape='ovo')
classifier.fit(normed_vectors, train_labels)

f = open(PICKLE, 'wb')
pickle.dump(classifier, f)
f.close()

with open(MEANS, "w") as f:
    for mean in means:
        f.write(str(mean) + '\n')

with open(STEDDEVS, "w") as f:
    for stddev in stddevs:
        f.write(str(stddev) + '\n')