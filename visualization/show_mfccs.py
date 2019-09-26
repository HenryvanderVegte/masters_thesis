import librosa.display
import numpy as np
import os
from global_vars import *
import matplotlib.pyplot as plt

audio_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//wavs//Ses03F_script01_1_M022.wav')

y, sr = librosa.load(audio_path)

mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

melspec = librosa.feature.melspectrogram(y=y, sr=sr)


plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(melspec,
                                             ref=np.max),
                         y_axis='mel', fmax=8000,
                         x_axis='time')
plt.colorbar(format='%+2.0f dB')
#plt.title('Mel spectrogram')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(ROOT_FOLDER, 'test//HRSes03F_script01_1_M022_ang.png'), dpi=300)