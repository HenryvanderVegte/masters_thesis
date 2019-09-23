from utils.experiments_util import *
from classification.util.global_vars import *

tsv_old = os.path.join(ROOT_FOLDER, 'datasets//MS//train.tsv')
tsv_out = os.path.join(ROOT_FOLDER, 'datasets//MS//train_new.tsv')

tsv_old_lines = open(tsv_old).read().splitlines()

tsv_new = 'Id\t' + tsv_old_lines[0] + '\n'

for i in range(1, len(tsv_old_lines)):
    tsv_new += str(i - 1) + '\t' + tsv_old_lines[i] + '\n'

with open(tsv_out, 'w') as f:
    f.write(tsv_new)
