from tqdm import tqdm
import glob
import numpy as np
import os
import sys
import tensorflow as tf
import librosa


"""

convert(image_paths=image_paths_train,
        out_path=path_tfrecords_train)

convert(image_paths=image_paths_test,
        out_path=path_tfrecords_test)
"""


def convert(speech_path, out_path, sound_format='wav', sample_rate=44100, mono=True):
    # Args:
    # wav_paths   List of file-paths for the speech files (train or test folder path)
    # out_path    File-path for the TFRecords output file.
    emo_dict = {'angry':0, 'fearful':1, 'happy':2, 'sad':3, 'calm':4}

    print("Converting: " + out_path)
    speech_paths = glob.glob(speech_path + "/*/*." + sound_format)
    

    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:        
        # Iterate over all the speech-paths and class-labels.
        for path in tqdm(speech_paths):
            # Load the speech-file using Librosa.
            speech, sr = librosa.load(path, res_type='kaiser_best', sr=sample_rate, mono=mono)
            print("\nspeech.shape: " + str(speech.shape))
            speech_list = speech.tolist()
            print("\nspeech_list: " + str(len(speech)))
            
            # Convert sound to raw bytes being speech a numpy.ndarray of float32 data.
            speech_bytes = speech.tostring()
            print("\nspeech_bytes lenght: " + str(len(speech_bytes)))

            mfccs = np.mean(librosa.feature.mfcc(y=speech, sr=sample_rate, n_mfcc=40).T,axis=0)
            mfccs_list = mfccs.tolist()
            
            # Extract the label
            # Labels: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Neutral
            label = int(path.split("/")[-2])
            print(path)
            print(label)

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'audio': tf.train.Feature(float_list=tf.train.FloatList(value=mfccs_list)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

if __name__=='__main__':
    sound_format='wav'
    sample_rate=44100
    mono=True
    convert('../data/train', '../data/train.record', sound_format, sample_rate, mono)
    convert('../data/test', '../data/test.record', sound_format, sample_rate, mono)
    print('Done')
