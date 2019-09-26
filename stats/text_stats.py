from global_vars import *
from utils.experiments_util import *
import collections

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
    print('Length\tCount')
    for key in orderedDict.keys():
        print(str(key) + '\t' + str(orderedDict[key]))

def print_word_count_distrib_grouped_by_impro_or_scripted(tsv_data):
    considered_labels = ['hap', 'exc', 'sad', 'ang', 'neu']
    count_map_impro = {}
    count_map_scripted = {}
    for instance in tsv_data:
        if instance['Label'] not in considered_labels:
            continue

        word_count = int(instance['Word_Count'])

        if 'impro' in instance['Name']:
            if word_count in count_map_impro:
                count_map_impro[word_count] = count_map_impro[word_count] + 1
            else:
                count_map_impro[word_count] = 1
        else:
            if word_count in count_map_scripted:
                count_map_scripted[word_count] = count_map_scripted[word_count] + 1
            else:
                count_map_scripted[word_count] = 1

    orderedImproDict = collections.OrderedDict(sorted(count_map_impro.items()))
    print('Impro Counts:')
    print('Length\tCount')
    for key in orderedImproDict.keys():
        print(str(key) + '\t' + str(orderedImproDict[key]))

    orderedScriptedDict = collections.OrderedDict(sorted(count_map_scripted.items()))
    print('Scripted counts:')
    print('Length\tCount')
    for key in orderedScriptedDict.keys():
        print(str(key) + '\t' + str(orderedScriptedDict[key]))

tsv_data = read_tsv_metadata(os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels.tsv"))
print_word_count_distrib_grouped_by_impro_or_scripted(tsv_data)