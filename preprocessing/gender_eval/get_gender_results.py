from utils.experiments_util import *
from preprocessing.emobase.emobase_feature_extractor import *

def print_m_f_statistics(gender_info_file):
    with open(gender_info_file, "r") as f:

        m_predicted = []
        m_gold = []
        f_predicted = []
        f_gold = []

        lines = f.readlines()
        for line in lines:
            split = line.split('\t')
            predicted = int(split[0])
            gold = int(split[1])
            utterance_name = split[2]

            if utterance_name.split('_')[-1].startswith('F'):
                f_predicted.append(predicted)
                f_gold.append(gold)
            else:
                m_predicted.append(predicted)
                m_gold.append(gold)

        print('Male stats: ')
        print(get_metrics_str(m_gold, m_predicted))

        print('Female stats: ')
        print(get_metrics_str(f_gold, f_predicted))


gender_info_file = os.path.join(ROOT_FOLDER, 'gender//audio_utterance_svm.txt')
print('AUDIO:')
print_m_f_statistics(gender_info_file)

#gender_info_file = os.path.join(ROOT_FOLDER, 'gender//text_classifier.txt')
#print('TEXT:')
#print_m_f_statistics(gender_info_file)

#gender_info_file = os.path.join(ROOT_FOLDER, 'gender//combined_classifier.txt')
#print('COMBINED:')
#print_m_f_statistics(gender_info_file)
