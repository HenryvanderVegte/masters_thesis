from global_vars import *
from utils.experiments_util import *

full_feature_file = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_with_pauses.npy')
result_feature_file = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_with_pauses_top_256.npy')
feature_ranking = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobaseFeaturesBestToWorst.txt')

feature_ranking = open(feature_ranking).read().splitlines()

feature_ranking = [int(i) for i in feature_ranking]

exlude_features = feature_ranking[256:]

full_feature_dict = np.load(full_feature_file).item()

result_feature_dict = exclude_sequence_features_by_id(full_feature_dict, exlude_features)

np.save(result_feature_file, result_feature_dict)