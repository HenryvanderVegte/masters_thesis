import os
from classification.util.global_vars import *
import numpy as np

wav_folder = os.path.join(ROOT_FOLDER, "VOXCELEB")
ids = os.path.join(ROOT_FOLDER, "VOXCELEB//ids.txt")
names = os.path.join(ROOT_FOLDER, "VOXCELEB//names.txt")
sett = os.path.join(ROOT_FOLDER, "VOXCELEB//set.txt")
track = os.path.join(ROOT_FOLDER, "VOXCELEB//track.txt")
sp = os.path.join(ROOT_FOLDER, "VOXCELEB//sp.txt")
denseFrameWavIds = os.path.join(ROOT_FOLDER, "VOXCELEB//denseFrameWavIds.txt")
wavLogits = os.path.join(ROOT_FOLDER, "VOXCELEB//wavLogits.txt")

def read_lines(file):
    file_in = open(file, "r")
    file_in_lines = file_in.readlines()
    lines = []
    for line in file_in_lines[:5]:
        info = line[:-1]
        lines.append(info)
        print(info)

def read_label_lines(file):
    file_in = open(file, "r")
    file_in_lines = file_in.readlines()
    lines = []
    for line in file_in_lines[:300]:
        highest_probs = []
        labels_dists = line[:-1].split('\t')[:-1]

        for label_dist in labels_dists:
            label_dist = label_dist.split(";")[:-1]
            label_dist = np.array([float(i) for i in label_dist])
            highest_probs.append(np.argmax(label_dist))
        lines.append(highest_probs)
    print(lines)

ids = read_lines(ids)
names = read_lines(names)
sett = read_lines(sett)
track = read_lines(track)
sp = read_lines(sp)
denseFrameWavIds = read_lines(denseFrameWavIds)
labels = read_label_lines(wavLogits)


