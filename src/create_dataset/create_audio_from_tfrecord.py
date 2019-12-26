"""
This script needs to be modified assuming that the "audio" field of the TFRecord file contains the mfccs instead of a sequence of the audio bytes in hex string format. 
You should pay attention to the following links to reconstruct an audio file from the mfccs features:

https://github.com/librosa/librosa/issues/424

https://github.com/librosa/librosa/issues/660
"""

import tensorflow as tf
import numpy as np
import os
import librosa
import scipy

sound_format = 'wav'
sample_rate = 44100
AUDIO_WIDHT = 285
AUDIO_HEIGHT = 40
AUDIO_SHAPE = 120423

def convert(in_path, audio_dir):
    # Args:
    # in_path   File-path for the TFRecords output file.
    # audio_dir      Folder to leave the generated audios.

    audio_index = {}
    
    if os.path.exists(in_path):
        print("Converting: " + in_path)
        audio_set = os.path.splitext(os.path.basename(in_path))[0]
    else:
        print(in_path + " not found!")
        return -1

    audio_iterator = tf.python_io.tf_record_iterator(path=in_path)

    for audio_record in audio_iterator:

        example = tf.train.Example()
        example.ParseFromString(audio_record)

        audio = example.features.feature["audio"].float_list.value
        label = example.features.feature["label"].int64_list.value[0]
        
        try:
            audio_index[label] += 1
        except KeyError:
            audio_index[label] = 1

        curr_index = audio_index[label]        

        audio_path = os.path.join(audio_dir, audio_set, str(label))
        if not os.path.exists(audio_path):
            os.makedirs(audio_path)	
        
        print(type(audio))
        audio_np = np.array(audio, dtype=np.float32)
        print(audio_np.shape)
        librosa.output.write_wav(os.path.join(audio_path, str(label) + "_" + str(curr_index) + ".wav"), audio_np, sr=sample_rate)
        
        


if __name__=='__main__':
    convert('../data/train.record', '../data_vis/')
    print('Done')
