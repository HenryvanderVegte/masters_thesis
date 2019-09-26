from global_vars import *
from utils.experiments_util import *

metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
labels = ["hap", "exc", "sad", "ang","neu"]

def get_all_info_str_from_CV(experiment_path):
    """
    Returns a dictionary with Utterance Name as key with tuples (Predicted, Gold)
    :param experiment_path:
    :return:
    """
    all_results = {}
    for i in range(0,10):
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

def is_total_agreement(all_labels_str):
    lab = all_labels_str.split(';')[0]
    total_agreement = True
    for label in all_labels_str.split(';'):
        if label != lab:
            total_agreement = False
    return total_agreement

def get_stats(experiment_path, metadata):
    experiment_info = get_all_info_str_from_CV(experiment_path)

    correct_with_full_agree = 0
    correct_without_full_agree = 0
    wrong_with_full_agree = 0
    wrong_without_full_agree = 0
    for instance in metadata:
        if instance['Name'] not in experiment_info:
            continue

        classified_correct = experiment_info[instance['Name']][0] ==experiment_info[instance['Name']][1]
        total_agreement = is_total_agreement(instance['Labels_All'])

        if classified_correct:
            if total_agreement:
                correct_with_full_agree += 1
            else:
                correct_without_full_agree += 1
        else:
            if total_agreement:
                wrong_with_full_agree += 1
            else:
                wrong_without_full_agree += 1

    print('Classified correct and full agreement:' + str(correct_with_full_agree))
    print('Classified correct and no full agreement:' + str(correct_without_full_agree))
    print('Classified wrong and full agreement:' + str(wrong_with_full_agree))
    print('Classified wrong and no full agreement:' + str(wrong_without_full_agree))

experiment_path = os.path.join(ROOT_FOLDER, "models//CV//12//CV_late_fusion_joined_model_output_svm")
get_stats(experiment_path, metadata)

