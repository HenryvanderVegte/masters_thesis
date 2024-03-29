from utils.experiments_util import *
from global_vars import *
from preprocessing.openSMILE.emobase_feature_extractor import *
from multiprocessing.pool import ThreadPool

metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
features_out = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_utterance_level.npy')

target_labels = ['hap', 'exc', 'sad', 'ang', 'neu']
names = []
for instance in metadata:
    if instance['Label'] in target_labels:
        names.append(instance['Name'])

pool = ThreadPool(8)
results = pool.map(extract_utterance_level_emobase_features, names)
pool.close()
pool.join()

instance_dict = {}
for i, name in enumerate(names):
    if len(results[i]) is 0:
        print('No results for ' + name)
        continue
    res = np.array(results[i])
    instance_dict[name] = res

np.save(features_out, instance_dict)
