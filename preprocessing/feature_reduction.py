from global_vars import *
from utils.experiments_util import *
from collections import Counter

def create_top_n_features_file(full_feature_file_path, result_feature_file_path, feature_ranking_path, count):
    feature_ranking = open(feature_ranking_path).read().splitlines()
    feature_ranking = [int(i) for i in feature_ranking]

    exlude_features = feature_ranking[count:]

    full_feature_dict = np.load(full_feature_file_path).item()

    result_feature_dict = exclude_sequence_features_by_id(full_feature_dict, exlude_features)

    np.save(result_feature_file_path, result_feature_dict)

'''
full_feature_file = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_with_pauses.npy')
result_feature_file = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_with_pauses_top_256.npy')
feature_ranking = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobaseFeaturesBestToWorst.txt')
create_top_n_features_file(full_feature_file, result_feature_file, feature_ranking, 256)
'''


def print_top_n_feature_names(feature_ranking_path, feature_description_path, count):
    feature_ranking = open(feature_ranking_path).read().splitlines()
    feature_description = open(feature_description_path).read().splitlines()

    prefixes = []
    for id in feature_ranking[:count]:
        line = feature_description[int(id)]

        pref1 = line.split(' ')[1].split('_')[0]
        pref2 = line.split(' ')[1].split('_')[1]
        pref2 = pref2.split('[')[0]
        prefix = pref1 + '_' + pref2
        prefixes.append(prefix)
        print(feature_description[int(id)])

    counter = Counter()
    counter.update(prefixes)
    print(counter)

'''
bestToWorst = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobaseFeaturesBestToWorst.txt')
feature_description = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_features.txt')
print_top_n_feature_names(bestToWorst, feature_description, 32)
'''
