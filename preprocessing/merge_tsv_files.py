from global_vars import *
from utils.experiments_util import *

iemocap_meta = read_tsv_metadata(os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels.tsv"))
ravdess_meta = read_tsv_metadata(os.path.join(ROOT_FOLDER, "datasets//RAVDESS//labels.tsv"))
savee_meta = read_tsv_metadata(os.path.join(ROOT_FOLDER, "datasets//SAVEE//labels.tsv"))
out_tsv = os.path.join(ROOT_FOLDER, "datasets//labels.tsv")



out_full = "Id\tName\tLabel\tSet\tDataset\tAudio\n"
id = 0

for instance in iemocap_meta:
    out_full += str(id) + "\t" + instance['Name'] + "\t" + instance['Label']  + "\t" + instance['Set']  + "\t" + 'IEMOCAP' + "\t" + 'IEMOCAP\\' + instance['Audio'] +"\n"
    id += 1

for instance in ravdess_meta:
    out_full += str(id) + "\t" +  instance['Name'] + "\t" + instance['Label']  + "\t" + instance['Set']  + "\t" + 'RAVDESS' + "\t" + 'RAVDESS\\' + instance['Audio'] +"\n"
    id += 1

for instance in savee_meta:
    out_full += str(id) + "\t" +  instance['Name'] + "\t" + instance['Label']  + "\t" + instance['Set']  + "\t" + 'SAVEE' + "\t" + 'SAVEE\\' + instance['Audio'] +"\n"
    id += 1

with open(out_tsv, "w") as f:
    f.write(out_full)
