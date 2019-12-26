# To run this script:
# 1) cd $HOME/catkin_ws/src/rz_se/src/trainer
# 2) Edit sound_path in line: emo_sound_recognizer_test(sound_path, params, endpoint)
# 3) python predict_epistemic_unc.py --job_dir ../experiments/ebayes_keras 

import os
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
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('--job_dir', default='../experiments/base_model/', help="Model path")
parser.add_argument('--pred_mode', default='local', help="local / cloud")
parser.add_argument('--epistemic_mc_sim', default=10, help='Epistemic Monte Carlo simulations')
parser.add_argument('--data_dir', default='../data/',
                    help="Directory containing the dataset (Train and Test TFRecords)")

args = parser.parse_args()



def mfccs_to_emo(mfccs, emodel, params_file, st):
    keras_input_names_list = 'input_1'
    #emo = estimator.predict(input_fn=model.pred_input_fn(mfccs, params_file, keras_input_names_list))
    print("\nmfccs: ")
    print(mfccs)
    inp_pred = model.input_fn_pred(mfccs, params_file, keras_input_names_list)
    inp_pred = tf.Print(inp_pred, [inp_pred], "\nInp_pred: ", summarize=100)
    emo = emodel.predict(inp_pred, steps=1)
    #emo = emodel.predict(model.input_fn_pred(mfccs, params_file, keras_input_names_list), steps=1)
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
        """
        emo_gen = mfccs_to_emo(mfccs, model, params_file, st)
        print("\nemo_gen: ")
        print(emo_gen)

        #emo = next(emo_gen)["epistemic_softmax_mean"]
        #epis_var = next(emo_gen)["epistemic_variance"]
        
        epis_var = next(emo_gen)["logits_variance"][-1]
        emo = next(emo_gen)["softmax_output"]
        """
        emo_list = mfccs_to_emo(mfccs, model, params_file, st)
        print("\nemo_list: ")
        print(emo_list)
        epis_var = emo_list[0][0][-1] #  we take variance from logits_variance
        print("\nepistemic variance: ")
        print(epis_var)
        emo = emo_list[1] # softmax_output
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
    return str(emotion), res, emo, epis_var



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
    str_emo, int_emo, softmax_emo, epis_var  = what_emo(mfccs, model, params_file, st, endpoint)
    return str_emo, int_emo, softmax_emo, epis_var
    


if __name__ == "__main__":
    gcp_project = "dh-dia4a" 
    gcp_name = "rz_se"
    gcp_version = "v1"
    endpoint = ""

    json_path = os.path.join(args.job_dir, "params.json")
    params = utils.Params(json_path)
    """
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    save_checkpoints_steps=10000, 
                                    save_checkpoints_secs=None,
                                    keep_checkpoint_max = 3,
                                    save_summary_steps=params.save_summary_steps)
    """   
    tf.keras.backend.set_learning_phase(1) # Train mode which will activate the dropout layers
    print("\ntf.keras.backend.learning_phase(): " + str(tf.keras.backend.learning_phase()))
    keras_input_names_list = 'input_1'
    print("\nLoading model ...")

    if args.pred_mode == "local":
        epis_model = model.create_bayesian_model(params.mfcc_size, params.num_classes)
        print("\nCompiling Bayesian Aleatoric Model. We're going to get the Epistemic Uncertainty ...\n")
        epis_model.compile(optimizer=tf.keras.optimizers.Adam(), loss={'logits_variance': model.bayesian_categorical_crossentropy(params.num_mc_simulations, params.num_classes), 'softmax_output': tf.keras.losses.categorical_crossentropy}, metrics={'softmax_output': 'accuracy'}, loss_weights={'logits_variance': 0.2, 'softmax_output': 1.})

        for layer in epis_model.layers:
            print("\nLayer name: " + str(layer.name)) 
        #loaded_estimator = tf.keras.estimator.model_to_estimator(keras_model=epis_model, model_dir=args.job_dir, config=config)
       
        
        latest = tf.train.latest_checkpoint(args.job_dir)
        print("\nWe're going to load weights from: " + str(latest))
        if latest != None:
            print("\nLoading weights from: " + str(latest))
            try:
                epis_model.load_weights(latest)
            except:
                print("\nThe model hasn't loaded the weights")
        
    else:
        endpoint = 'projects/{}/models/{}'.format(gcp_project, gcp_name)
        if gcp_version is not None:
            endpoint += '/versions/{}'.format(gcp_version)

    str_epis_mc_preds = []
    int_epis_mc_preds = []
    softmax_epis_preds = []
    epis_var_preds = []
    #emo_sound_recognizer_test('raw_1553515100.36.wav', loaded_estimator, params, endpoint)
    
    for i in range(args.epistemic_mc_sim): 
        #tf.keras.backend.set_learning_phase(1)
        print("\ntf.keras.backend.learning_phase(): " + str(tf.keras.backend.learning_phase()))
        str_emotion, int_emotion, softmax_emotion, epis_var = emo_sound_recognizer_test('raw_1553507047.18.wav', epis_model, params, endpoint)
        print('\nThe predicted emotion of the input audio for the Monte Carlo simulation ' + str(i) + ' is: ' + str(str_emotion))
        str_epis_mc_preds.append(str_emotion)
        int_epis_mc_preds.append(int_emotion)
        softmax_epis_preds.append(softmax_emotion)
        epis_var_preds.append(epis_var)

    print("\nlen(softmax_epis_preds): " + str(len(softmax_epis_preds)) + "\n")
    print("\nlen(softmax_epis_preds[0]): " + str(len(softmax_epis_preds[0])) + "\n")
    #print(softmax_epis_preds)
    np_softmax_epis_preds = np.array(softmax_epis_preds)
    print("\nSample's Softmax values of each of the " + str(args.epistemic_mc_sim) + " MC simulations (np_softmax_epis_preds): ")
    print(np_softmax_epis_preds)
    print("\nStd of the softmax values of the " + str(args.epistemic_mc_sim) + " MC simulations (np_softmax_epis_preds.std(axis=0)) using the input sample: ")
    print(np_softmax_epis_preds.std(axis=0))

    np_epis_var_preds = np.array(epis_var_preds)
    print("\nSample's variance of each of the " + str(args.epistemic_mc_sim) + " MC simulations (np_epis_var_preds): ")
    print(np_epis_var_preds)
    print("\nStd of the variances of the " + str(args.epistemic_mc_sim) + " MC simulations (np_epis_var_preds.std(axis=0)) using the input sample: ")
    print(np_epis_var_preds.std(axis=0))
    print("\nMean of the variances of the " + str(args.epistemic_mc_sim) + " MC simulations (np_epis_var_preds.mean(axis=0)) using the input sample: ")
    print(np_epis_var_preds.mean(axis=0))
    
    
    # probability + variance
    print("\nClasses --> 0: angry, 1: fearful, 2: happy, 3: sad, 4: calm")
    print("\nThis sample belongs to: ")
    for i, (prob, var) in enumerate(zip(np_softmax_epis_preds.mean(axis=0), np_softmax_epis_preds.std(axis=0))):
        print("\nClass: " + str(np.argmax(prob)) + " with probability: " + str(prob) + "\n\nand variance: " + str(var))
        #print("class: {}; with probability: {:.1%}; and variance: {:.13%} ".format(i, prob, var))

    


