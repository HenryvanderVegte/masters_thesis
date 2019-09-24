from ms_dataset.ms_util import *
from global_vars import *
from utils.rnn_utils import *

train_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//train.tsv'))
dev_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//dev.tsv'))
test_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//test.tsv'))

dev_names = set()
for instance in dev_metadata:
    dev_names.add(instance['mRawDataId'])
print(len(dev_names))

test_names = set()
for instance in test_metadata:
    test_names.add(instance['mRawDataId'])
print(len(test_names))

for instance in train_metadata:
    dataid = instance['mRawDataId']
    if dataid in test_names or dataid in dev_names:
        print(dataid)
