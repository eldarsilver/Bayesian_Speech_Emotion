import librosa
import numpy as np
import glob
import os
from tqdm import tqdm


class SpeechAugmentation:
    def read_file(self, finpath, sample_rate=44100):
        speech, sr = librosa.load(finpath, res_type='kaiser_best', sr=sample_rate, mono=True)
        return speech

    def add_noise(self, speech, noise_amount):
        noise = np.random.randn(len(speech))
        speech_noise = speech + noise_amount * noise
        return speech_noise

    def shift(self, speech):
        return np.roll(speech, 1600)

    def write_file(self, foutpath, speech, sample_rate=44100):
        librosa.output.write_wav(foutpath, speech, sample_rate)


# Set the amount of noise
noise_amount = 0.2

# Set index emotion from 0 to 4
index_emo = '0'

# Set input folder
finpaths = os.path.join('../data/train', index_emo)

# Set output folder
fopaths = os.path.join('../data/train_sa', index_emo)

# Set the sound format
sound_format = 'wav'

sa = SpeechAugmentation()
speech_paths = glob.glob(os.path.join(finpaths, '*.' + sound_format))

# Iterate over all the speech-paths and class-labels. 
for i, path in enumerate(speech_paths):
    # Load the speech-file using Librosa.
    print("\nReading " + str(path))
    speech = sa.read_file(path)
    # Add noise
    speech_noise = sa.add_noise(speech, noise_amount)
    # Write noise speech
    fout = os.path.join(fopaths, str(i) + '.' + sound_format)
    print("\nWriting " + str(fout))
    sa.write_file(fout, speech_noise) 
    

