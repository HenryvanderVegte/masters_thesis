import nltk
from collections import Counter
EXTRACTED_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//extracted_data_skip_labels_remove_nonverbal.txt"

text_file = open(EXTRACTED_FILE, "r")
lines = text_file.readlines()

token_counts = []
for line in lines[1:]:
    split = line.split('\t')
    if len(split) != 3:
        print(line)
    utterance = split[2]
    tokens = nltk.word_tokenize(utterance)
    token_counts.append(len(tokens))

print(Counter(token_counts))

for key, value in sorted(Counter(token_counts).items()):
    print(str(key) + ';' + str(value))