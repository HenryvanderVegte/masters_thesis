from utils.experiments_util import *
from stats.results.result_utils import get_all_info_str_from_CV
from global_vars import *

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
