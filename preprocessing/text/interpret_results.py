import os
from classification.util.global_vars import *
from old.word_embeddings import extract_tokens_from_utterance

class_groups = {
    "0":"hxc",
    "1":"sad",
    "2":"ang",
    "3":"neu"
}

dev_txt = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//utterances//dev.txt")

results_file = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//results_interpretation//results-interpretation.txt"
results_file_c = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//results_interpretation//results-interpretation3.txt"

file_in = open(dev_txt, "r")
file_in_lines = file_in.readlines()

name_to_utt = {}
for line in file_in_lines:
    split = line.split("\t")

    name = split[0]
    utterance = split[2][:-1]
    tokens = extract_tokens_from_utterance(utterance)
    cleaned_utterance = " ".join(tokens)
    name_to_utt[name] = cleaned_utterance



file_in = open(results_file, "r")
file_in_lines = file_in.readlines()

res_corr = ""
res_wrong = ""
for line in file_in_lines:
    split = line[:-1].split("\t")
    pred = split[0]
    gold = split[1]
    name = split[2]

    if pred == gold:
        res_corr += class_groups[pred] + "\t" + class_groups[gold] + "\t" + name + "\t" + name_to_utt[name] + "\n"
    else:
        res_wrong += class_groups[pred] + "\t" + class_groups[gold] + "\t" + name + "\t" + name_to_utt[name] + "\n"

with open(results_file_c, "w") as f:
    f.write(res_wrong)
    f.write("\n\n")
    f.write(res_corr)