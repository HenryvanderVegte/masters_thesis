from utils.result_utils import get_all_info_str_from_CV
from utils.experiments_util import *

def print_gender_seperated_statistics(experiment_path):
    all_info = get_all_info_str_from_CV(experiment_path)

    m_predicted = []
    m_gold = []
    f_predicted = []
    f_gold = []

    for name in all_info.keys():
        predicted = int(all_info[name][0])
        gold = int(all_info[name][1])

        if name.split('_')[-1].startswith('F'):
            f_predicted.append(predicted)
            f_gold.append(gold)
        elif name.split('_')[-1].startswith('M'):
            m_predicted.append(predicted)
            m_gold.append(gold)
        else:
            print('Unexpected Name')

    print('Male stats (' + str(len(m_gold)) + ' instances)')
    print(get_metrics_str(m_gold, m_predicted))

    print('Female stats (' + str(len(f_gold)) + ' instances)')
    print(get_metrics_str(f_gold, f_predicted))

def is_total_agreement(all_labels_str):
    lab = all_labels_str.split(';')[0]
    total_agreement = True
    for label in all_labels_str.split(';'):
        if label != lab:
            total_agreement = False
    return total_agreement

def print_agreement_seperated_statistics(experiment_path, metadata):
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