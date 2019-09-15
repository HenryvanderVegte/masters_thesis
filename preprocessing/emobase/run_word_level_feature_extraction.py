from utils.experiments_util import *
from preprocessing.emobase.emobase_feature_extractor import *
from multiprocessing.pool import ThreadPool

wltfile = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//dev.txt')
features_out = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_dev.npy')

lines = open(wltfile).read().splitlines()[:1]

pool = ThreadPool(8)
results = pool.map(extract_word_level_emobase_features, lines)
pool.close()
pool.join()

instance_dict = {}
for i, line in enumerate(lines):
    name = line.split('\t')[0]
    if len(results[i]) is 0:
        print('No results for ' + name)
        continue
    res = np.array(results[i])
    instance_dict[name] = res

np.save(features_out, instance_dict)
