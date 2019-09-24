from utils.experiments_util import *
from global_vars import *

train0 = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train0.npy')
train1 = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train1.npy')
train2 = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train2.npy')
train3 = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train3.npy')
train4 = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train4.npy')
train5 = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train5.npy')

train0 = np.load(train0).item()
train1 = np.load(train1).item()
train2 = np.load(train2).item()
train3 = np.load(train3).item()
train4 = np.load(train4).item()
train5 = np.load(train5).item()

new = {**train0, **train1, **train2, **train3, **train4, **train5}


new_path = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train.npy')
np.save(new_path, new)


