import os
from classification.util.global_vars import *
from shutil import copyfile

iemocap_full = 'E://masters_thesis//datasets//IEMOCAP_full_release_withoutVideos//IEMOCAP_full_release'
out_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//features//forced_alignment")

for r, d, f in os.walk(iemocap_full):
    for file in f:
        if '.wdseg' in file:
            base_path = os.path.basename(os.path.normpath(r))
            old_path = os.path.join(r, file)
            new_path = os.path.join(out_folder, file)
            copyfile(old_path, new_path)
