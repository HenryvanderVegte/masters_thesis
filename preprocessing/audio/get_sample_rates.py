import os
import pydub

WAV_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_wav"
for r, d, f in os.walk(WAV_FOLDER):
    for file in f:
        if '.wav' in file:
            path = os.path.join(r, file)
            audio = pydub.AudioSegment.from_wav(path)
            if str(audio.frame_rate) != "16000":
                print(audio.frame_rate)

