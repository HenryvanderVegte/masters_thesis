import math

def get_oovs_latex_format(oov_words):
    oov_words.sort()
    elements_per_row = 4
    rows = math.ceil((len(oov_words) / elements_per_row))

    lines = ['' for i in range(rows)]

    for i in range(len(oov_words)):
        lines[i % rows] = lines[i % rows] + '&' + oov_words[i]

    for line in lines:
        line = line[1:] + '\\\\'
        print(line)