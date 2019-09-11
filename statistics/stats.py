import os
import collections
from classification.util.global_vars import *
from utils.experiments_util import *

tsv_data = read_tsv_metadata(os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels.tsv"))


def print_word_count_distrib(tsv_data):
    considered_labels = ['hap', 'exc', 'sad', 'ang', 'neu']
    count_map = {}
    for instance in tsv_data:
        if instance['Label'] not in considered_labels:
            continue

        word_count = int(instance['Word_Count'])

        if word_count in count_map:
            count_map[word_count] = count_map[word_count] + 1
        else:
            count_map[word_count] = 1

    orderedDict = collections.OrderedDict(sorted(count_map.items()))
    for key in orderedDict.keys():
        print(str(key) + ';' + str(orderedDict[key]))

print_word_count_distrib(tsv_data)