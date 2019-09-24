from global_vars import *
import os
from random import shuffle

nr_of_folds = 10

labels_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels_no_folds.tsv')
labels_with_fold_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels_new.tsv')

new_file_str = ''

lines = open(labels_path).read().splitlines()

new_file_str += lines[0] + '\tFold\n'

sessions = []
for row in lines[1:]:
    session = row.split('\t')[1][:-5]
    if session not in sessions:
        sessions.append(session)

shuffle(sessions)

session_to_fold = {}
for i, session in enumerate(sessions):
    session_to_fold[session] = i % nr_of_folds

for row in lines[1:]:
    session = row.split('\t')[1][:-5]
    new_file_str += row + '\t' + str(session_to_fold[session]) + '\n'

with open(labels_with_fold_path, 'w') as f:
    f.write(new_file_str)