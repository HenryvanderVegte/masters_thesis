import numpy as np
from utils.experiments_util import get_metrics_str

def test(labels, feature_vectors, logger):
    logger.info("############ Testing max. probability classifier. ########## \n\n" )

    labels = np.array(labels)
    features_count = feature_vectors.shape[1]
    cut_index = int(features_count/2)
    first_half = feature_vectors[:,:cut_index]
    second_half = feature_vectors[:,cut_index:]
    summed = first_half + second_half

    predictions = np.argmax(summed, axis=1)

    predictions = [str(i) for i in predictions]
    labels = [str(i) for i in labels]

    logger.info(get_metrics_str(labels, predictions))