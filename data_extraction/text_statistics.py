from collections import Counter

EXTRACTED_FILE = "C://Users//Henry//Desktop//IEMOCAP_txt//split//test.txt"

def print_labels(file_lines):
    labels = []
    # skip first line containing comments
    for line in file_lines[1:]:
        labels.append(line.split('\t')[1])
    counter = Counter(labels)
    print(counter)



text_file = open(EXTRACTED_FILE, "r")
lines = text_file.readlines()

print_labels(lines)