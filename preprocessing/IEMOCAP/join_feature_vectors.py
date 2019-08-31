from classification.util.global_vars import *
from utils.experiments_util import *
from classification.text.word_embeddings import *
import os

metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))

audio_features = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_50ms_buffer.npy')
word_embedding_features = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//text//embeddings//2d//word_embeddings.npy')

def join_feature_vectors(vecs1, vecs2):
    """
    Gets two lists of vectors and creates a joined list out of it.
    For instance:
    vecs1 = [ [0, 1], [2, 3] ]
    vecs2 = [ [4], [5] ]
    would return
    joined_vecs = [ [0, 1, 4], [2, 3, 5] ]
    :param vecs1:
    :param vecs2:
    :return:
    """
    if len(vecs1) != len(vecs2) or len(vecs1) == 0:
        print('Invalid number of feature vectors: ' + str(len(vecs1)) + ', ' + str(len(vecs2)))
        return

    number_of_vecs = len(vecs1)
    vec1_length = len(vecs1[0])
    vec2_length = len(vecs2[0])
    joined_vecs = np.zeros(shape=(number_of_vecs, vec1_length + vec2_length))

    for i in range(len(vecs1)):
        joined_vecs[i, 0:vec1_length] = vecs1[i]
        joined_vecs[i, vec1_length:vec1_length+vec2_length] = vecs2[i]
    return joined_vecs

def join_featuresets(featureset1, featureset2):
    """
    Gets two dictionaries of featuresets, joins feature vectors based on the dictionary keys and returns the joined dictionary
    :param featureset1:
    :param featureset2:
    :return:
    """
    joined_instances = {}
    names = []
    for audio_name in featureset1.keys():
        if audio_name in featureset2:
            names.append(audio_name)
    for name in names:
        joined_vec = join_feature_vectors(featureset1[name], featureset2[name])
        joined_instances[name] = joined_vec
    return joined_instances


audio_features = np.load(audio_features).item()
word_embedding_features = np.load(word_embedding_features).item()

joined_fs = join_featuresets(audio_features, word_embedding_features)

out = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//late_fusion//emobase_50ms_buffer_and_embeddings.npy')
np.save(out, joined_fs)