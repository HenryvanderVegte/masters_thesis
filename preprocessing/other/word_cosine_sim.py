import numpy as np
import gensim
import os
from scipy import spatial

model = gensim.models.KeyedVectors.load_word2vec_format('file://C://Users//Henry//Desktop//Masterarbeit//pretrained_embeddings//GoogleNews-vectors-negative300.bin', binary=True)


hate = np.array(model.wv['hate'])
hell = np.array(model.wv['hell'])

fun = np.array(model.wv['fun'])
excited = np.array(model.wv['excited'])

print(1 - spatial.distance.cosine(hate, hell))
print(1 - spatial.distance.cosine(hate, fun))
print(1 - spatial.distance.cosine(hate, excited))
print(1 - spatial.distance.cosine(hell, fun))
print(1 - spatial.distance.cosine(hell, excited))
print(1 - spatial.distance.cosine(fun, excited))