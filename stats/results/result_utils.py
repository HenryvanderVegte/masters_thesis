import os

def get_all_info_str_from_CV(experiment_path, nr_of_folds = 10):
    """
    Returns a dictionary with Utterance Name as key with tuples (Predicted, Gold)
    :param experiment_path:
    :return:
    """
    all_results = {}
    for i in range(0,nr_of_folds):
        results = os.path.join(experiment_path, str(i), 'results.txt')
        reached_res = False
        for line in open(results).read().splitlines():
            if line.startswith('Predicted	Gold	Name'):
                reached_res = True
                continue
            if not reached_res:
                continue

            split = line.split('\t')
            all_results[split[2]] = (split[0], split[1])
    return all_results