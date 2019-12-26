# To run this script:
# 1) cd $HOME/catkin_ws/src/rz_se/src/trainer
# 2) Edit sound_path in line: emo_sound_recognizer_test(sound_path, params, endpoint)
# 3) python predict.py

import base64
import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import model
import utils
import argparse
import datetime
import time
import googleapiclient.discovery
import librosa


parser = argparse.ArgumentParser()
parser.add_argument('--job_dir', default='../experiments/base_model/', help="Model path")
parser.add_argument('--pred_mode', default='local', help="local / cloud")
args = parser.parse_args()



def mfccs_to_emo(mfccs, estimator, params_file, st):
    keras_input_names_list = 'input_1'
    emo = estimator.predict(input_fn=model.pred_input_fn(mfccs, params_file, keras_input_names_list))
    return emo


def what_emo(mfccs, model, params_file, st, endpoint):
    """
    Implements emotion recognition based on the audio.
    
    Arguments:
    mfccs -- numpy array with the mfccs of the audio
    model -- your model instance in Keras
    
    Returns:
    
    emotion -- string, the emotion prediction for the audio
    """
    print("\nEntering what_emo... ")
    emotion = None
    emo_label = {0:'angry', 1:'fearful', 2:'happy', 3:'sad', 4:'calm'}
    if endpoint == "":
        emo_gen = mfccs_to_emo(mfccs, model, params_file, st)
        emo = next(emo_gen)["dense_1"]
        #emo = next(emo_gen)
        print("\nemo: ")
        print(emo)
    else:
        dict_im = {}
        service = googleapiclient.discovery.build('ml', 'v1')
        dict_im["input_1"] = mfccs.tolist()
        predictions = service.projects().predict(name=endpoint, body={'instances':[dict_im]}).execute()["predictions"]
        emo = np.array(predictions[0]['dense_1'])   
        print("\nemo: ")
        print(emo)
    res = np.argmax(emo)
    print("\nres: ")
    print(res)
    emotion = emo_label[res]    
    print("\nPrediction: " + str(emotion))
    return str(emotion)



def emo_sound_recognizer_test(sound_path, model, params_file, endpoint):
    print("\nEntering emo_sound_recognizer_test... ")
    if params_file.mono == 1:
        mono = True
    else:
        mono = False
    numpy_speech, sr = librosa.load(sound_path, res_type='kaiser_best', sr=params_file.sample_rate, mono=mono)
    mfccs = np.mean(librosa.feature.mfcc(y=numpy_speech, sr=params_file.sample_rate, n_mfcc=params_file.mfcc_size).T,axis=0)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    emo = what_emo(mfccs, model, params_file, st, endpoint)
    print('Audio saved to ../audios_recog/' + str(emo) + "_" + str(st) + '.wav')
    #cv2.imwrite('../audios_recog/' + str(emo) + "_" + str(st) + '.wav', face)  



if __name__ == "__main__":
    gcp_project = "dh-dia4a" 
    gcp_name = "rz_se"
    gcp_version = "v1"
    endpoint = ""

    json_path = os.path.join(args.job_dir, "params.json")
    params = utils.Params(json_path)
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    save_checkpoints_steps=10000, 
                                    save_checkpoints_secs=None,
                                    keep_checkpoint_max = 3,
                                    save_summary_steps=params.save_summary_steps)




    cnn_model = model.conv_model(params.mfcc_size, params.num_classes)
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    for layer in cnn_model.layers:
        print("\nLayer name: " + str(layer.name)) 
    keras_input_names_list = 'input_1'
    print("\nLoading model ...")
    if args.pred_mode == "local":
        loaded_estimator = tf.keras.estimator.model_to_estimator(keras_model=cnn_model, model_dir=args.job_dir, config=config)
    else:
        endpoint = 'projects/{}/models/{}'.format(gcp_project, gcp_name)
        if gcp_version is not None:
            endpoint += '/versions/{}'.format(gcp_version) 
    emo_sound_recognizer_test('raw_1553514401.4.wav', loaded_estimator, params, endpoint)
  
