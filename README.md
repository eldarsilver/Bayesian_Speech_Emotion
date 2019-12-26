# CREATE DATASET<br>
## CAPTURE AUDIOS<br>
* Firstly, audios must be captured using the robot microphone. The folder structure for the training set should have as many subfolders as emotions need to be detected, so that each subfolder will store the audios of the same emotion. This rule has to be applied to create the structure of test subfolders. <br>
* Turn on the robot, go to the next step while becomes up and running.<br>
* Open Terminal1:<br>
    * Load Anaconda environment:<br>
        * `$ source activate environment`<br>
    * Load ROS environment:<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
    * Once the robot is up and running:
        * `$ cd $HOME/catkin_ws/src/mic2wav/launch/`<br>
        * `$ gedit mic2wav_params.yaml`<br>
        * There are several things to look at:<br>
            * The topic where the robot publishes the raw audio it captures is already set (`/pepper_robot/audio`). If another device is going to be used, the topic can be set changing the value of the parameter called `raw_audio_topic`.<br>
            * The destination folder can be changed using the  `sound_path` parameter. Each recording session should be done repeating several sentences, using the same voice emotion. After that it will depend on the tagger to place the audio recording made in the train or test subfolder corresponding to the emotion used during the recording.<br>
            * There are other parameters that can be changed, for instance `dest_num_channels` which is the number of channels for the desired conversion, `dest_rate` which is the rate for the desired conversion or `max_iter` which sets the number of iterations after which the sound will be dumped into a WAV file.<br>
            * Once we're done with the tweaking, let's capture some audios!<br>
    * `$ roslaunch mic2wav mic2wav.launch`<br>
    * At the end of the process there should be as many folders as emotions. All these folders under the same parent folder.<br>
    * Now it's time to split all the audios (the original and the distorted ones) into TRAIN/TEST set.<br>

## GET TFRECORDS
* Once data has been split into TRAIN and TEST, it's time to transform it into TFRecord format.
    * `$ cd $HOME/catkin_ws/src/rz_se/src/create_dataset`
    * `$ gedit create_tfrecord.py`:
        * We can define in the next to last line the origin directory, the destination directory, the sound format of the input audios, their sample rate and if they are  mono or stereo. All of them are parameters of the `convert` function.
    * `$ python create_tfrecord.py`
* As a result, the data from the origin directory will be transformed into a TFRecord format file with the specified name and in the specified directory.

# TRAIN AND TEST THE MODEL LOCALLY<br>
* `$ cd $HOME/catkin_ws/src/rz_se/src`<br>
* Here can be found `task.py`. It is worth having a look at the parameters of this script:
    * `--config_dir` should point to the directory where the `params.json` is stored. This file contains model configuration options.
    * `--data_dir` should point to the directory where the train and test TFRecords are stored.
    * `--job-dir` should point to the directory where previous checkpoints are located or where the training checkpoints will be stored.
* If you want to train the model locally using the Network defined in the function def conv_model in the model.py script:<br>
    * `python task.py --config_dir /home/datahack/catkin_ws/src/rz_se/src/experiments/base_model/ --job-dir /home/datahack/catkin_ws/src/rz_se/src/experiments/base_model/ --data_dir /home/datahack/catkin_ws/src/rz_se/src/data/`<br>
* If you want to test the model locally using the model trained with the function def conv_model in the model.py script:<br>
    * `python task.py --config_dir /home/datahack/catkin_ws/src/rz_se/src/experiments/base_model/ --job-dir /home/datahack/catkin_ws/src/rz_se/src/experiments/base_model/ --data_dir /home/datahack/catkin_ws/src/rz_se/src/data/ --mode test`<br>
* Optionally open Terminal2:
    * Go to the directory set in `--job-dir`.
    * `$ tensorboard --logdir=.`
    * This allows to check how the training progresses.

# TRAIN THE MODEL IN GOOGLE CLOUD PLATFORM (GCP)<br>
* It's assumed that a dataset consisting of two tfrecord format files (train.record and test.record), has been created by following previous steps.<br>
* Open Terminal1:<br>
    * Create bucket (preferably in the same zone where the computation is done):<br>
        * `$ gsutil mb -l us-east1 gs://rz_se`<br>
    * Copy the train.record file to the bucket (the "data" directory is created automatically):<br>
        * `$ gsutil cp train.record gs://rz_se/data/`<br>
    * Copy the test.record file to the bucket:<br>
        * `$ gsutil cp test.record gs://rz_se/data/`<br>
    * Package the project:<br>
        * `$ python $HOME/catkin_ws/src/rz_se/src/setup.py sdist`<br>
    * Train the job in GCP (using 1 GPU):<br>
        * `$ gcloud ml-engine jobs submit training rz_se_1 --package-path $HOME/catkin_ws/src/rz_se/src/trainer --module-name trainer.task --region us-east1 --job-dir gs://rz_se/output --scale-tier BASIC_GPU --runtime-version 1.8 -- --data_dir gs://rz_se/data/ --conv_model True`<br>
        * Note that the job name must be changed each time a submission occurs (/rz_se2`, /rz_se_3`...)<br>
    * Train the job in GCP (using multiple GPUs. You should create a config.yaml file like https://cloud.google.com/ml-engine/docs/tensorflow/machine-types ):<br>
        * `$ gcloud ml-engine jobs submit training rz_se_1 --package-path $HOME/catkin_ws/src/rz_se/src/trainer --module-name trainer.task --region us-east1 --job-dir gs://rz_se/output --config $HOME/catkin_ws/src/rz_se/src/config.yaml --runtime-version 1.8 -- --data_dir gs://rz_se/data/  --conv_model True`<br>
* Open Terminal2:<br>
    * `$ tensorboard --logdir=gs://rz_se/output`<br>
* Open Terminal3:<br>
    * Inspect the /rz_se_1` (or whichever the job name is) job's logs as they arrive:<br>
        * `$ gcloud ml-engine jobs stream-logs rz_se_1`
    * Cancel job rz_se_1` if it's necessary:<br>
        * `$ gcloud ml-engine jobs cancel rz_se_1`
    * Again if needed, the bucket can be deleted issuing the following:<br>
        * $ `gsutil rb gs://rz_se/`

# PREDICTION WITH DATA FROM THE ROBOT
## LOCAL MODE
* Here it will be showed how to locally run the model to cast predictions from the robot data.<br>
* Go to `$HOME/catkin_ws/src/rz_se/launch/rz_se_params.yaml` and ensure `model_path` points to the local directory where its graph and checkpoints are stored.<br>
* In the same file `pred_mode` must be set to local.<br>
* Open Terminal1:<br>
    * Edit BICA State Machine so that the BICA ROS component to test is configured as a dependency.<br>
    * `$ gedit $HOME/catkin_ws/src/datahack_host/bica_examples/src/Diafora_executor.cpp`<br>
        * Search and edit `Terapia_code_once()` method.<br>
        * Look for the `addDependency` instruction and replace its parameter by `rz_se_bica`.<br>
        * The `Terapia_code_once()`should look like this:<br>
            * `void Diafora_executor::Terapia_code_once()`<br>
              `{`<br>
                 `ROS_INFO("TERAPIA: CODE ONCE");`<br>
                 `addDependency("rz_se_bica");`<br>
              `}`<br>
        * For these changes to become effective, it's suggested to (carefully) proceed as follows:<br>
            * `$ cd $HOME/catkin_ws/`<br>
            * `$ rm -rf build/ devel/`<br>
            * `$ catkin_make`
    * Now all the packages have been rebuilt and every change will become effective. Let's activate the speech2text service:
        * Log into the robot:<br>
            * `$ ssh nao@<ROBOT_IP>`<br>
        * Load ROS environment:<br>
            * `$ source System/setup.bash`<br>
        * Launch Google Speech2Text Node:<br>
            * `$ roslaunch pepper_dialog speech2text_google.launch`<br>
* Open Terminal2:<br>
    * Load Anaconda environment:<br>
        * `$ source activate environment`<br>
    * Load ROS environment:<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
    * Launch rz_se BICA ROS node:<br>
        * `$ roslaunch rz_se rz_se.launch`<br>
* Open Terminal3:<br>
    * Let's inspect the speech2text topic so that we have a clue about how the robot recognizes the voice commands we're issuing:<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
    * The topic to track corresponds to the parameter s2t_topic from `$HOME/catkin_ws/src/rz_se/launch/rz_se_params.yaml` (by default `/pepper_utils/hri_comm`)<br>
        * `$ rostopic echo <s2t_topic>` (as explained, it should be `rostopic echo /pepper_utils/hri_comm`)<br>
* Open Terminal4:<br>
    * Let's inspect the `audio_emotions_topic` just to see how the detected voice emotions are output:<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
    * The topic to track corresponds to the parameter `audio_emotions_topic` from `$HOME/catkin_ws/src/rz_se/launch/rz_se_params.yaml` (by default `/pepper_utils/audio_emotion`)<br>
        * `$ rostopic echo <audio_emotions_topic>` (as explained, it should be `rostopic echo /pepper_utils/audio_emotion`)<br>
* Open Terminal5:<br>
    * Let's run the BICA state machine:<br>
        * Load Anaconda environment:<br>
            * `$ source activate environment`<br>
        * Load ROS environment:<br>
            * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
            * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
        * Run, run, run!<br>
            * `$ rosrun bica_examples Diafora_executor_node`<br>
* Try it!<br>
    * In order to activate the recognition issue the command *reconoce*<br>
        * From this moment on, the model will receive audio through Pepper's microphone.<br>
    * To stop recognition you can issue the command *termina*<br>
    * Apart from topics, results can be checked both at `$HOME/catkin_ws/src/rz_se/src/audio_recog/emotions.txt` and `$HOME/catkin_ws/src/rz_se/src/audio_recog/` where each sound detected is going to be saved as a *.wav* audio file labeled with the following format: *"raw_" + str(emo) + "_" + timestamp + ".wav"* (i.e. *raw_Happy_timestamp.wav*) <br>

## GOOGLE CLOUD PLATFORM (GCP) MODE<br>
### GENERATE MODEL/VERSION FOR A GOOGLE CLOUD PLATFORM (GCP) TRAINED MODEL<br>
* Open Terminal1:<br>
    * First we have to locate the bucket and directory where `saved_model.pb` has been stored. Let's assume ours is: `gs://rz_se/output/1553518004/`<br>
    * If it's a new model, it has, firstly, to be created in GCP (where /rz_se* is the model's name):<br>
        * Note that the specified region should match the bucket region.<br>
        * `$ gcloud ml-engine models create rz_se --regions=us-east1`<br>
    * Create model's version in GCP (where --origin is the directory where save_model.pb is stored):<br>
        * `$ gcloud ml-engine versions create v1 --model rz_se --origin gs://rz_se/output/1553518004/ --runtime-version 1.8`<br>
    * Additionally it is possible to display existing models and their default versions:<br>
        * `$ gcloud ml-engine models list`<br>
    * And also delete version (The default version can't be deleted unless there aren't more versions):<br>
        * `$ gcloud ml-engine versions delete v1 --model rz_se`<br>

### GENERATE MODEL/VERSION FOR A LOCALLY TRAINED MODEL<br>
*  Open Terminal1:<br>
    * Firstly we have to take care of generating the cloud required `saved_model.pb` file.<br>
    * To do this, the graph should be exported. Let's see how:<br>
        * `$ cd $HOME/catkin_ws/src/rz_se/src/trainer`<br>
        * `python task.py --mode only_export --job-dir ../experiments/base_model/ --export-dir ../experiments/base_model/export/exporter/ --export-graph`<br>
        * As a result of this, the `saved_model.pb` file (and possibly a `variables` directory) will be created at the location pointed in the --export-dir flag.<br>
        * Create bucket to allocate the `saved_model.pb` (and the `variables` directory)<br>
            * `$ gsutil mb -l us-east1 gs://rz_se`<br>
        * Copy `saved_model.pb` and the variables directory (if exists) inside the bucket<br>
            * `$ cd /home/datahack/catkin_ws/src/rz_se/src/experiments/base_model/export/exporter/`
            * `$ gsutil cp -r <export_directory>/* gs://rz_se/output/`<br>
            * Sometimes the gsutil copy command may play you a trick...ensure that `saved_model.pb` and `variables` directory (if exists) have been properly copied, both at the same hierarchy level.<br>
        * Let's create the model:
            * If it's a new model, it has, firstly, to be created in GCP (where/rz_se is the model's name):<br>
                * Note that the specified region should match the bucket region.<br>
                * `$ gcloud ml-engine models create rz_se --regions=us-east1`<br>
            * Create model's version in GCP (where --origin is the directory where save_model.pb is stored):<br>
                * `$ gcloud ml-engine versions create v1 --model rz_se --origin gs://rz_se/output/1553518004/ --runtime-version 1.8`<br>
            * Additionally it is possible to display existing models and their default versions:<br>
                * `$ gcloud ml-engine models list`<br>
            * And also delete version (The default version can't be deleted unless there aren't more versions):<br>
                * `$ gcloud ml-engine versions delete v1 --model rz_se`<br>

### CONFIGURE PREDICTION FOR BOTH GOOGLE CLOUD PLATFORM (GCP) AND LOCALLY TRAINED MODELS<br>
* Here it will be showed how to use the model just created in Google Cloud ML to cast predictions from the robot data.<br>
* Go to `$HOME/catkin_ws/src/rz_se/launch/rz_se_params.yaml` and check the following parameters are correctly set:<br>
    * `gcp_name` should be set to the model's name (according to the previous steps: `rz_se`)
    * `gcp_project` should be set to the project's name (dh-dia4a)
    * `gcp_version` should be set to the model's version to be used (v1, v2...)
* In the same file `pred_mode` must be set to cloud.<br>
* Open Terminal1:<br>
    * Edit BICA State Machine so that the BICA ROS component to test is configured as a dependency.<br>
    * `$ gedit $HOME/catkin_ws/src/datahack_host/bica_examples/src/Diafora_executor.cpp`<br>
        * Search and edit `Terapia_code_once()` method.<br>
        * Look for the `addDependency` instruction and replace its parameter by /rz_se_bica`.<br>
        * The `Terapia_code_once()`should look like this:<br>
            * `void Diafora_executor::Terapia_code_once()`<br>
               `{`<br>
                 `ROS_INFO("TERAPIA: CODE ONCE");`<br>
                 `addDependency("rz_se_bica");`<br>
               `}`<br>
        * For these changes to become effective, it's suggested to (carefully) proceed as follows:<br>
            * `$ cd $HOME/catkin_ws/`<br>
            * `$ rm -rf build/ devel/`<br>
            * `$ catkin_make`
    * Now all the packages have been rebuilt and every change will become effective. Let's activate the speech2text service:
        * Log into the robot:<br>
            * `$ ssh nao@<ROBOT_IP>`<br>
        * Load ROS environment:<br>
            * `$ source System/setup.bash`<br>
        * Launch Google Speech2Text Node:<br>
            * `$ roslaunch pepper_dialog speech2text_google.launch`<br>
* Open Terminal2:<br>
    * Load Anaconda environment:<br>
        * `$ source activate environment`<br>
    * Load ROS environment:<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
    * Launch re_tfoda BICA ROS node:<br>
        * `$ roslaunch rz_se rz_se.launch`<br>
* Open Terminal3:<br>
    * Let's inspect the speech2text topic so that we have a clue about how the robot recognizes the voice commands we're issuing:<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
    * The topic to track corresponds to the parameter s2t_topic from `$HOME/catkin_ws/src/rz_se/launch/rz_se_params.yaml` (by default `/pepper_utils/hri_comm`)<br>
        * `$ rostopic echo <s2t_topic>` (as explained, it should be `rostopic echo /pepper_utils/hri_comm`)<br>
* Open Terminal4:<br>
    * Let's inspect the emotions_topic just to see how the detected face emotions are output:<br>
        * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
        * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
    * The topic to track corresponds to the *audio_emotions_topic* from `$HOME/catkin_ws/src/rz_se/launch/rz_se_params.yaml` (by default `/pepper_utils/audio_emotion`)<br>
        * `$ rostopic echo <audio_emotions_topic>` (as explained, it should be `rostopic echo /pepper_utils/audio_emotion`)<br>
* Open Terminal5:<br>
    * Let's run the BICA state machine:<br>
        * Load Anaconda environment:<br>
            * `$ source activate environment`<br>
        * Load ROS environment:<br>
            * `$ source $HOME/catkin_ws/devel/setup.bash`<br>
            * `$ export ROS_MASTER_URI=http://<ROBOT_IP>:11311`<br>
        * Run, run, run!<br>
            * `$ rosrun bica_examples Diafora_executor_node`<br>
* Try it!<br>
    * In order to activate the recognition issue the command *reconoce*:<br>
        * From this moment on, the model will receive images through Pepper's camera.<br>
    * To stop recognition you can issue the command *termina*.<br>
    * Apart from topics, results in the form of images can be checked at `$HOME/catkin_ws/src/rz_se/src/audio_recog/emotions.txt` and `$HOME/catkin_ws/src/rz_se/src/faces_recog/` where each sound detected is going to be saved as a *.wav* audio file labeled with the following format: *"raw_" + str(emo) + "_" + timestamp + ".wav"* (i.e. *raw_Happy_timestamp.wav*) <br>