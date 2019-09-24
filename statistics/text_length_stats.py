import os
from global_vars import *

txt_files = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment//utterances")

lengths_impro = []
lengths_script = []

for r, d, f in os.walk(txt_files):
    for file in f:
        file_path = os.path.join(r, file)
        with open(file_path, "r") as txt:
            lines = txt.readlines()
            if 'impro' in file:
                lengths_impro.append(len(lines))
            else:
                lengths_script.append(len(lines))

print('Impro lines:' + str(len(lengths_impro)))
print('Script lines:' + str(len(lengths_script)))

print('----')
for e in lengths_impro:
    print(e)
print('----')
for e in lengths_script:
    print(e)