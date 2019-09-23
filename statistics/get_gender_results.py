from utils.experiments_util import *
from classification.util.global_vars import *

def print_m_f_statistics(gender_infos):
    m_predicted = []
    m_gold = []
    f_predicted = []
    f_gold = []
    all_predicted = []
    all_gold = []

    for line in gender_infos:
        split = line.split('\t')
        predicted = int(split[0])
        gold = int(split[1])
        utterance_name = split[2]

        if utterance_name.split('_')[-1].startswith('F'):
            f_predicted.append(predicted)
            f_gold.append(gold)
        elif utterance_name.split('_')[-1].startswith('M'):
            m_predicted.append(predicted)
            m_gold.append(gold)
        else:
            print('Unexpected Name')
        all_predicted.append(predicted)
        all_gold.append(gold)

    print('Male stats: ')
    print('Count: ' + str(len(m_gold)))
    print(get_metrics_str(m_gold, m_predicted))

    print('Female stats: ')
    print('Count: ' + str(len(f_gold)))
    print(get_metrics_str(f_gold, f_predicted))

    #print('Total stats: ')
    #print('Count: ' + str(len(all_gold)))
    #print(get_metrics_str(all_gold, all_predicted))

def get_gender_info_str_from_CV(experiment_path):
    all_results = []
    for i in range(0,10):
        results = os.path.join(experiment_path, str(i), 'results.txt')
        reached_res = False
        for line in open(results).read().splitlines():
            if line.startswith('Predicted	Gold	Name'):
                reached_res = True
                continue
            if not reached_res:
                continue
            all_results.append(line)
    return all_results


experiment_path = os.path.join(ROOT_FOLDER, "models//CV//12//CV_late_fusion_joined_model_output_svm")
gender_info_as_str = get_gender_info_str_from_CV(experiment_path)
print_m_f_statistics(gender_info_as_str)

