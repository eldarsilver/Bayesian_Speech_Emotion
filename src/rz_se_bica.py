#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
import base64
import cv2
import glob
import googleapiclient.discovery
import numpy as np
import os
import rospy
import threading
import time
import tensorflow as tf
import trainer.model
import trainer.utils
from bica_core.bica_core import Component
import datetime
from std_msgs.msg import String
import alsaaudio
import librosa
import struct
from naoqi_bridge_msgs.msg import AudioBuffer
from subprocess import call



class seComponent(Component):
    def __init__(self, name):
        print("Entering constructor rz_se_bica")
        Component.__init__(self, name)
 
      

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server. See ../launch/rz_se_params.yaml
        """      
        audio_emotions_topic = rospy.get_param("~audio_emotions_topic")
        gcp_name = rospy.get_param("~gcp_name")
        gcp_project = rospy.get_param("~gcp_project")
        gcp_version = rospy.get_param("~gcp_version")
        json_path = rospy.get_param("~json_path")
        model_path = rospy.get_param("~model_path")
        emotions_logfile = rospy.get_param("~emotions_logfile")
        robot_ip = rospy.get_param("~robot_IP")
        s2t_topic = rospy.get_param("~s2t_topic")
        pred_mode = rospy.get_param("~pred_mode")
        raw_audio_topic = rospy.get_param("~raw_audio_topic")
        dest_num_channels = rospy.get_param("~dest_num_channels")
        dest_rate = rospy.get_param("~dest_rate")
        max_iter = rospy.get_param("~max_iter")     
        sound_path = rospy.get_param("~sound_path")
        wav_topic = rospy.get_param("~wav_topic")
        return (audio_emotions_topic, gcp_name, gcp_project, gcp_version, json_path, model_path, emotions_logfile, robot_ip, s2t_topic, pred_mode, raw_audio_topic, dest_num_channels, dest_rate, max_iter, sound_path, wav_topic)



    def pred_input_fn(self, image, params_file, keras_input_names_list):
        x_pred = tf.convert_to_tensor(image, dtype=tf.float32)
        x_pred = x_pred / 255.0
        x_pred = tf.reshape(x_pred, (-1, params_file.image_size, params_file.image_size, params_file.num_channels))
        dataset = tf.data.Dataset.from_tensor_slices(x_pred)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        #x = {keras_input_names_list: x_train}
        x = iterator.get_next()
        return x


    def genHeader(self, sampleRate, bitsPerSample, channels, samples):
        datasize = len(samples) * channels * bitsPerSample // 8
        #datasize = len(samples) * bitsPerSample // 8
        o = bytes("RIFF").encode('ascii')                                      # (4byte) Marks file as RIFF
        #o += to_bytes(datasize + 36, 4,'little')                              # (4byte) File size in bytes excluding this and RIFF marker
        o += struct.pack('<I',datasize + 36)                                   # (4byte) File size in bytes excluding this and RIFF marker
        o += bytes("WAVE").encode('ascii')
        o += bytes("fmt ").encode('ascii')                                     # (4byte) Format Chunk Marker
        #o += to_bytes(16, 4,'little')                                         # (4byte) Length of above format data
        o += struct.pack('<I',16)                                              # (4byte) Length of above format data                     
        #o += to_bytes(1, 2,'little')                                          # (2byte) Format type (1 - PCM)
        o += struct.pack('<H',1)                                               # (2byte) Format type (1 - PCM)
        #o += to_bytes(channels, 2,'little')                                   # (2byte)
        o += struct.pack('<H',channels)                                        # (2byte)
        #o += to_bytes(sampleRate, 4,'little')                                 # (4byte)
        o += struct.pack('<I', sampleRate)                                     # (4byte)
        #o += to_bytes((sampleRate * channels * bitsPerSample) // 8, 4,'little')# (4byte)
        o += struct.pack('<I',(sampleRate * channels * bitsPerSample) // 8)    # (4byte)
        #o += to_bytes((channels * bitsPerSample) // 8, 2,'little')            # (2byte)
        o += struct.pack('<H', (channels * bitsPerSample) // 8)                # (2byte)
        #o += to_bytes(bitsPerSample, 2,'little')                              # (2byte)
        o += struct.pack('<H', bitsPerSample)                                  # (2byte)
        o += bytes("data").encode('ascii')                                     # (4byte) Data Chunk Marker
        #o += to_bytes(datasize, 4,'little')                                    # (4byte) Data size in bytes
        o += struct.pack('<I',datasize)                                    # (4byte) Data size in bytes

        return o



    def emospeechCB(self, msg):
        rospy.loginfo("Callback received!")
        print(len(msg.channelMap))
        print(msg.frequency)
        print(len(msg.data))
        # Number of channels and frequency are set
        self.device.setchannels(len(msg.channelMap))
        self.device.setrate(msg.frequency)
        # The current audio sequence is processed and appended to what's gathered in dataBuff so far
        tmp = np.array(list(msg.data)).reshape(-1, 4)
        if (self.dataBuff is not None):
            self.dataBuff = np.vstack((self.dataBuff, tmp))
        else:
            self.dataBuff = tmp                
        print(self.counter)
        # Only when max_iter is reached, the audio processing occurs
        if (self.counter == self.max_iter):
            print(self.dataBuff.shape)
            print(len(self.dataBuff))
            header = self.genHeader(48000, 16, 4, self.dataBuff)
            body = ""
            bodyElements = None
            channels = range(4)
            for sample in self.dataBuff:                
                for c in channels:     
                    bodyChannel = []
                    s = sample[c]
                    sf = float(s) / float(32768)
                    if(sf>1):
                        sf = 1
                    if(sf<-1):
                        sf = -1
                    bodyChannel.append(sf)
                    body += struct.pack('<h', int(sf*32767))           
            #print("Saving array")
            self.dataBuff = np.true_divide(self.dataBuff.astype(np.float32), float(32768))
            self.dataBuff = np.where(self.dataBuff < -1.0 , -1.0, self.dataBuff)
            self.dataBuff = np.where(self.dataBuff > 1.0, 1.0, self.dataBuff)  
            self.dataBuff = np.mean(self.dataBuff, axis=1)
            maxvalue = np.amax(self.dataBuff)
            print("\ndataBuff Max Value: " + str(maxvalue))
            print("\nself.dataBuff.shape: ")
            print(self.dataBuff.shape)
            #np.save(os.path.join(self.sound_path,"nparray" + timestamp + ".npy"), self.dataBuff)
            if maxvalue > 0.2:
                self.last_header = header
                self.last_body = body
                self.last_mfccs = np.mean(librosa.feature.mfcc(y=self.dataBuff, sr=self.params.sample_rate, n_mfcc=self.params.mfcc_size).T,axis=0)                   
            else:
                self.last_mfccs = None
            self.dataBuff = None
            self.counter = 0
        else:
            self.counter += 1



    def speechCB(self, msg):
        endCB = False
        # Speech to text result is retrieved
        speech = msg.data        
        print("Entering speechCB")
        # On activation command
        try:
            #speech.index("reconoce")
            if not endCB and not self.recognizing:
                print("Reconociendo...")
                print(self.activate)
                if self.activate == True:
                    print("Subscribing to: " + str(self.raw_audio_topic))
                    # Let's subscribe to the topic where the robot publishes audio.
                    self.sub_sound = rospy.Subscriber(self.raw_audio_topic, AudioBuffer, self.emospeechCB, queue_size=1) 
                    self.recognizing = True
                    endCB = True
        except ValueError:
            pass
        # On deactivation command
        try:
            speech.index("termina")
            if not endCB and self.recognizing:
                #Unsubscribe from the topic where the robot publishes audio.
                print("Unsubscribing from: " + str(self.raw_audio_topic))
                self.sub_sound.unregister()
                self.recognizing = False
                endCB = True
        except ValueError:
            pass



    def mfccs_to_emo(self, mfccs, estimator, params_file, st):
        emo = estimator.predict(input_fn=trainer.model.pred_input_fn(mfccs, params_file, self.keras_input_names_list))
        return emo



    def what_emo(self, mfccs, model, params_file, st, endpoint):
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
            emo_gen = self.mfccs_to_emo(mfccs, model, params_file, st)
            emo = next(emo_gen)["dense_1"]
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



    def step(self):
        print("P: step()")
        if self.last_mfccs is not None:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            emo = self.what_emo(self.last_mfccs, self.loaded_estimator, self.params, st, self.endpoint)
            print("\nlen(emo): " + str(len(emo)))
            print("\nemo value: " + str(emo))
            if len(emo) > 0:
                print("\nself.last_emo: " + str(self.last_emo))
                if self.last_emo != emo:
                    print("EMOTION CHANGE DETECTED")
                    print("ULTIMO " + str(self.last_emo))
                    print("ACTUAL " + str(emo))
                    self.last_emo = emo
                    self.pub_audio_emo.publish(emo)
                    timestamp = str(time.time())
                    filename_raw = os.path.join(self.sound_path,"raw_" + str(emo) + "_" + timestamp + ".wav")  
                    if (self.last_header != None) and (self.last_body != None):          
                        f = open(filename_raw, 'wb')
                        f.write(self.last_header + self.last_body)
                        f.close()
                        self.pub_wav_name.publish(self.last_header + self.last_body)
                        log_str = "The emotion of this audio " + str(filename_raw) + " is: " + str(emo) + "\n"
                        print(log_str)
                        self.log_emotions.write(log_str)                 
        self.last_mfccs = None
        self.last_header = None
        self.last_body = None



    def initClient(self):
        print("Entering initClient")
        # To control the number of iterations
        self.counter = 0
        print(self.counter)
        # To accumulate the incoming bytes for each iteration
        self.dataBuff = None
        
        self.last_mfccs = None
        self.endpoint = ""

        # Parameters are loaded from ../launch/mic2wav_params.yaml
        (self.audio_emotions_topic, gcp_name, gcp_project, gcp_version, json_path, model_path, emotions_logfile, robot_ip, s2t_topic, pred_mode, self.raw_audio_topic, self.dest_num_channels, self.dest_rate, self.max_iter, self.sound_path, self.wav_topic) = self.get_parameters()

        # Device is configured
        rospy.loginfo("Getting audio card...")
        self.device = alsaaudio.PCM()
        self.device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        rospy.loginfo("Done!")

        # Check whether the dir to store the processed sound exists
        if not (os.path.exists(self.sound_path)):
            os.makedirs(self.sound_path)

        # The name of the file containing the processed sound is published
        self.pub_wav_name = rospy.Publisher(self.wav_topic, String, queue_size=1)

        # The name of the topic where the audio emotion is going to be published
        self.pub_audio_emo = rospy.Publisher(self.audio_emotions_topic, String, queue_size=1)   
        
        
        self.params = trainer.utils.Params(json_path)
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                    save_checkpoints_steps=10000, 
                                    save_checkpoints_secs=None,
                                    keep_checkpoint_max = 3,
                                    save_summary_steps=self.params.save_summary_steps)




        cnn_model = trainer.model.conv_model(self.params.mfcc_size, self.params.num_classes)
        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        for layer in cnn_model.layers:
            print("\nLayer name: " + str(layer.name)) 
        self.keras_input_names_list = 'input_1'
        print("\nLoading model ...")
        if pred_mode == "local":
            self.loaded_estimator = tf.keras.estimator.model_to_estimator(keras_model=cnn_model, model_dir=model_path, config=config)
        else:
            self.endpoint = 'projects/{}/models/{}'.format(gcp_project, gcp_name)
            if gcp_version is not None:
                self.endpoint += '/versions/{}'.format(gcp_version) 

        self.log_emotions = open(emotions_logfile, 'a+')
  
        self.activate = False
        print("Subscribing to " + s2t_topic)
        # Subscribe to the topic where the Speech2Text node has published
        #self.s2t = rospy.Subscriber(s2t_topic, String, self.speechCB, queue_size=1) 
        self.recognizing = False
        self.activate = True
        self.last_img = None
        self.last_emo = None
        # Let's subscribe to the topic where the robot publishes audio. Remove the following '#' when the model doesn't need to be activated using a voice command ...
        self.sub_sound = rospy.Subscriber(self.raw_audio_topic, AudioBuffer, self.emospeechCB, queue_size=1) 
        


    def activateCode(self):
        print("Entering activateCode rz_se_bica")
        self.initClient()
        self.t = threading.Thread(target=self.step)
        self.tid = self.t.ident
        self.t.start()



    def deActivateCode(self):
        print("Entering deActivateCode rz_se_bica")
        self.s2t.unregister()
        self.activate = False
       

if __name__ == '__main__':
    try:
        rospy.init_node('rz_se_bica', anonymous=False)
        seComp = seComponent(1)
        while seComp.ok():
            pass
    except rospy.ROSInterruptException:
        pass
