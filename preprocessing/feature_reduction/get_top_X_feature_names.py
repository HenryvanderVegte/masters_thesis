from utils.experiments_util import *
from global_vars import *
from collections import Counter

bestToWorst = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobaseFeaturesBestToWorst.txt')
feature_description = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_features.txt')


bestToWorst = open(bestToWorst).read().splitlines()
feature_description = open(feature_description).read().splitlines()

top_x = 256

prefixes = []
for id in bestToWorst[:top_x]:
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