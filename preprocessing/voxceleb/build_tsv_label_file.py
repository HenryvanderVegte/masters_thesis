import os
from classification.util.global_vars import *
from collections import Counter
import numpy as np

wav_folder = os.path.join(ROOT_FOLDER, "VOXCELEB")
ids = os.path.join(ROOT_FOLDER, "VOXCELEB//ids.txt")
names = os.path.join(ROOT_FOLDER, "VOXCELEB//names.txt")
sett = os.path.join(ROOT_FOLDER, "VOXCELEB//set.txt")
track = os.path.join(ROOT_FOLDER, "VOXCELEB//track.txt")
sp = os.path.join(ROOT_FOLDER, "VOXCELEB//sp.txt")
denseFramesWavIds = os.path.join(ROOT_FOLDER, "VOXCELEB//denseFramesWavIds.txt")
wavLogits = os.path.join(ROOT_FOLDER, "VOXCELEB//wavLogits.txt")
out_labels = os.path.join(ROOT_FOLDER, "VOXCELEB//labels.tsv")

def read_lines(file):
    file_in = open(file, "r")
    file_in_lines = file_in.readlines()
    lines = []
    for line in file_in_lines:
        info = line[:-1]
        lines.append(info)
    return lines

def read_label_lines(file):
    file_in = open(file, "r")
    file_in_lines = file_in.readlines()
    all_frame_preds = []
    lines = []
    most_freq_labels = []
    for line in file_in_lines:
        highest_probs = []
        labels_dists = line[:-1].split('\t')[:-1]

        for label_dist in labels_dists:
            label_dist = label_dist.split(";")[:-1]
            label_dist = np.array([float(i) for i in label_dist])
            highest_probs.append(np.argmax(label_dist))
            all_frame_preds.append(np.argmax(label_dist))
        lines.append(highest_probs)
        most_freq = max(set(highest_probs), key=highest_probs.count)
        most_freq_labels.append(most_freq)

    print(Counter(all_frame_preds))

    for key, value in sorted(Counter(all_frame_preds).items()):
        print(str(key) + ';' + str(value))

    return lines, most_freq_labels

labels, most_common = read_label_lines(wavLogits)

print(Counter(most_common))

for key, value in sorted(Counter(most_common).items()):
    print(str(key) + ';' + str(value))

ids = read_lines(ids)
names = read_lines(names)
sett = read_lines(sett)
track = read_lines(track)
sp = read_lines(sp)
denseFrameWavIds = read_lines(denseFramesWavIds)

tsv_string = "id\tname\tset\ttrack\tsp\tdenseFrameWavId\tmost_frequent\tlabels\n"

for i in range(len(ids)):
    tsv_string += ids[i] + "\t"
    tsv_string += names[i] + "\t"
    tsv_string += sett[i] + "\t"
    tsv_string += track[i] + "\t"
    tsv_string += sp[i] + "\t"
    tsv_string += denseFrameWavIds[i] + "\t"
    tsv_string += str(most_common[i]) + "\t"
    label_dist = ';'.join(str(x) for x in labels[i])
    tsv_string += label_dist + "\n"

#with open(out_labels, 'w') as d:
#    d.write(tsv_string)