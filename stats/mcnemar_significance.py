import os
from utils.result_utils import get_all_info_str_from_CV
from global_vars import *
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar(experiment_path1, experiment_path2):
    dict1 = get_all_info_str_from_CV(experiment_path1)
    dict2 = get_all_info_str_from_CV(experiment_path2)

    '''
    Counts for all instances:
    '''
    t1t2 = 0 #Correctly classified by experiment 1 & experiment 2
    t1f2 = 0 #Correctly classified by experiment 1 & wrong experiment 2
    f1t2 = 0 #...
    f1f2 = 0 #...
    for key in dict1:
        exp1correct = dict1[key][0] == dict1[key][1]
        exp2correct = dict2[key][0] == dict2[key][1]

        if exp1correct and exp2correct:
            t1t2 += 1
        elif exp1correct and not exp2correct:
            t1f2 += 1
        elif not exp1correct and exp2correct:
            f1t2 += 1
        elif not exp1correct and not exp2correct:
            f1f2 += 1

    table = [[t1t2, t1f2],
             [f1t2, f1f2]]

    print(table)

    result = mcnemar(table, exact=False)
    print('statistic=%.3f, p-value=%.4f' % (result.statistic, result.pvalue))

    if result.pvalue <= 0.001:
        print('***')
    elif result.pvalue < 0.01:
        print('**')
    elif result.pvalue < 0.05:
        print('*')

experiment_path1 = os.path.join(ROOT_FOLDER, "models//CV//12//CV_classify_joined_word_level_256_emo")
experiment_path2 = os.path.join(ROOT_FOLDER, "models//CV//12//CV_classify_word_level_fusion_joined_features")
run_mcnemar(experiment_path1, experiment_path2)
