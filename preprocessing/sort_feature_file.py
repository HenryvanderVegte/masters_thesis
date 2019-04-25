import os

FILE_PATH = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_70.txt"

file = open(FILE_PATH, "r")
lines = file.readlines()

id_dict = {}

for line in lines:
    split = line.split('\t', 1)
    id_dict[split[0]] = split[1]

sorted_file = FILE_PATH + ".sorted"
with open(sorted_file, 'w') as d:
    for key in sorted(id_dict):
        d.write(key + "\t" + id_dict[key])