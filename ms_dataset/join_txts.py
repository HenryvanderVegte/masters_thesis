from utils.experiments_util import *
from classification.util.global_vars import *

train0 = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train0.txt')
train1 = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train1.txt')
train2 = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train2.txt')
train3 = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train3.txt')
train4 = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train4.txt')
train5 = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train5.txt')


train_out = os.path.join(ROOT_FOLDER, 'datasets//MS//transcriptions//train.txt')

t0 = open(train0).read()
t1 = open(train1).read()
t2 = open(train2).read()
t3 = open(train3).read()
t4 = open(train4).read()
t5 = open(train5).read()


out_str = t0 + t1 + t2 +t3 +t4 +t5

with open(train_out, 'w') as f:
    f.write(out_str)

