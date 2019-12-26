"""Define the model."""
from __future__ import division, absolute_import, print_function
from builtins import *
import tensorflow as tf
import numpy  as np
import sys
#from attn_augconv import augmented_conv2d
#import keras

def parse(serialized):
    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    # We know the length of both fields. If not the tf.VarLenFeature could be used
    features = {'audio': tf.FixedLenFeature([mfcc_size], tf.float32), 'label': tf.FixedLenFeature([], tf.int64)}
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    """
    context_features = {'label': tf.FixedLenFeature([], dtype=tf.int64)}
    sequence_features = {'audio': tf.FixedLenSequenceFeature([], dtype=tf.float32)}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=serialized, sequence_features=sequence_features, context_features=context_features) 

    audio_raw = sequence_parsed['audio']
    """

    # Get the audio.
    audio_raw = parsed_example['audio']
    audio_raw = tf.cast(audio_raw, tf.float32)
    audio = tf.reshape(audio_raw, (mfcc_size, 1))
    # Decode the raw bytes so it becomes a tensor with type.
    #image = tf.decode_raw(image_raw, tf.uint8)
    # The type is now uint8 but we need it to be float.
    #image = tf.cast(image, tf.float32)
    # Normalize from [0, 255] to [0.0, 1.0]
    #image = image / 255.0
    # Reshape 
    #image = tf.reshape(image, (image_size, image_size, num_channels))
    #print(image.shape)
    # Get the label associated with the example.
    label = parsed_example['label']
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=num_classes)  
    # The image and label are now correct TensorFlow types.
    return audio, label

def input_fn(filename_tfrecords, params, num_examples, train, keras_input_names_list, bayes_unc):
    global num_classes, mfcc_size
    num_classes = params.num_classes
    mfcc_size = params.mfcc_size
    dataset = tf.data.TFRecordDataset(filenames=filename_tfrecords)
    dataset = dataset.map(parse)
    if train:
        #dataset = dataset.shuffle(num_examples) # whole dataset into the buffer
        num_repeat = params.num_epochs
    else:
        num_repeat = params.num_epochs
        #num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1) # make sure you always have one batch ready to serve
    
    iterator = dataset.make_one_shot_iterator()

    
    examples_batch, labels_batch = iterator.get_next() 
    labels_batch = tf.reshape(labels_batch, (-1, params.num_classes))
    
    
    x = {keras_input_names_list: examples_batch}
    y = labels_batch
    if bayes_unc == 'aleatoric' or bayes_unc == 'epistemic':
        #return x, (y, y)
        examples_batch = tf.Print(examples_batch, [examples_batch], "\nBatch of examples: ", summarize=4)
        return examples_batch, (y, y)
    else:
        return x, y
        
def input_fn_pred(numpy_speech, params, keras_input_names_list):
    x_pred = tf.convert_to_tensor(numpy_speech, dtype=tf.float32)
    #x_pred = tf.reshape(x_pred, (-1, params.mfcc_size, 1))
    x_pred = tf.reshape(x_pred, (-1, params.mfcc_size, 1))
    print("\nx_pred: ")
    print(x_pred)
    #x_pred = tf.reshape(x_pred, (params.mfcc_size, 1))
    x = {keras_input_names_list: x_pred}
    return x_pred
    

def train_input_fn(path_tfrecords_train, params, num_examples, keras_input_names_list, bayes_unc):
    return lambda: input_fn(path_tfrecords_train, params, num_examples, True, keras_input_names_list, bayes_unc)


def test_input_fn(path_tfrecords_test, params, keras_input_names_list, bayes_unc):
    return lambda: input_fn(path_tfrecords_test, params, 0, False, keras_input_names_list, bayes_unc)

def pred_input_fn(numpy_speech, params, keras_input_names_list):
    # numpy_speech: the audio should be opened using ... numpy_speech, sr = librosa.load(path, res_type='kaiser_best', sr=sample_rate, mono=mono)
    return lambda: input_fn_pred(numpy_speech, params, keras_input_names_list)


### Simple CNN model using Keras Functional API ###

def conv_model(num_features, num_classes):
    input_layer = tf.keras.layers.Input(shape=(num_features, 1))
    print("\ninput_layer.shape: " + str(input_layer.shape))
    #input_layer = tf.keras.layers.Input(shape=(img_size, img_size, num_chan))
    conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(input_layer)
    #conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation=tf.nn.relu)(conv_1)
    #avgpool_a = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(conv_2)
    dropout_a = tf.keras.layers.Dropout(0.1)(conv_1)
    avgpool_a = tf.keras.layers.MaxPooling1D(pool_size=(8))(dropout_a)
    conv_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(avgpool_a)
    dropout_b = tf.keras.layers.Dropout(0.1)(conv_2)
    flatten = tf.keras.layers.Flatten()(dropout_b)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(flatten)


    return tf.keras.Model(inputs=input_layer, outputs=flatten)


########################################################


### Create Custom Dropout Layer  #######################

class Dropout(tf.keras.layers.Layer):

    """Always-on dropout layer, i.e. does not respect the training flag
    set to true by model.fit but false by model.predict.
    Unlike tf.keras.layers.Dropout, this layer does not return input
    unchanged if training=true, but always randomly drops a fraction specified
    by self.size of the input nodes.
    """

    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate


    def call(self, inputs):
        return tf.nn.dropout(inputs, self.rate)


    def get_config(self):
        # Enables model.save and restoration using tf.keras.models.load_model
        config = super().get_config()
        config["rate"] = self.rate
        return config


########################################################


### Create Bayesian Model ##############################

def create_bayesian_model(num_features, num_classes):    
    input_layer = tf.keras.layers.Input(shape=(num_features, 1))
    print("\ninput_layer.shape: " + str(input_layer.shape))
    conv_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=(5), padding='same', activation=tf.nn.relu)(input_layer)
    print("\nconv_1.shape: " + str(conv_1.shape))
    conv_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(conv_1)
    dropout_a = tf.keras.layers.Dropout(rate=0.5)(conv_2, training=True)
    maxpool_a = tf.keras.layers.MaxPooling1D(pool_size=(8))(dropout_a)
    conv_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(maxpool_a)
    conv_4 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(conv_3)
    conv_5 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(conv_4)
    dropout_b = tf.keras.layers.Dropout(rate=0.5)(conv_5, training=True)
    conv_6 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(dropout_b)
    flatten = tf.keras.layers.Flatten()(conv_6)
    
    bn_1 = tf.keras.layers.BatchNormalization(name='post_basenet')(flatten)
    dropout_1 = tf.keras.layers.Dropout(rate=0.5)(bn_1, training=True)
    #dropout_1 = Dropout(rate=0.5)(bn_1)
    dense_1 = tf.keras.layers.Dense(500, activation='relu')(dropout_1)
    bn_2 = tf.keras.layers.BatchNormalization(name='bn_2')(dense_1)
    dropout_2 = tf.keras.layers.Dropout(rate=0.5)(bn_2, training=True)
    #dropout_2 = Dropout(rate=0.5)(bn_2)
    dense_2 = tf.keras.layers.Dense(100, activation='relu')(dropout_2)
    bn_3 = tf.keras.layers.BatchNormalization(name='bn_3')(dense_2)
    dropout_3 = tf.keras.layers.Dropout(rate=0.5)(bn_3, training=True)
    #dropout_3 = Dropout(rate=0.5)(bn_3)

    logits = tf.keras.layers.Dense(num_classes)(dropout_3)
    softmax_output = tf.keras.layers.Activation('softmax', name='softmax_output')(logits)

    pre_variance = tf.keras.layers.Dense(1)(dropout_3)
    variance = tf.keras.layers.Activation('softplus', name='variance')(pre_variance)

    logits_variance = tf.keras.layers.concatenate([logits, variance], name='logits_variance')

    return tf.keras.Model(inputs= input_layer, outputs=[logits_variance, softmax_output])


########################################################


### Create Bayesian Categorical Crossentropy Loss Function #####

def bayesian_categorical_crossentropy(num_mc_simulations, num_classes):
    # pred_var shape: (N, num_classes + 1)
    def b_internal(true_labels, pred_var):
        # std shape: (N, 1)
        std = tf.keras.backend.sqrt(pred_var[:, num_classes])
        # variance shape: (N, 1)
        variance = pred_var[:, num_classes]
        variance = tf.Print(variance, [variance], "\nVariance: ", summarize=100) # 
        # The loss function should minimize a variance which is less than infinite and therefore we're going to exponientiate the variance
        variance_depressor = tf.keras.backend.exp(variance) - tf.keras.backend.ones_like(variance)
        # pred shape: (N, num_classes)
        pred = pred_var[:, 0:num_classes]
        pred = tf.Print(pred, [pred], "\nPred: ", summarize=100)
        # undistorted_loss shape: (N,)
        #undistorted_loss = tf.keras.losses.categorical_crossentropy(true_labels, pred, from_logits=True)
        undistorted_loss = tf.keras.backend.categorical_crossentropy(true_labels, pred, from_logits=True)
        undistorted_loss = tf.Print(undistorted_loss, [undistorted_loss], "\nUndistorted_loss: ", summarize=8)
        #tf.print(undistorted_loss, output_stream="file:///home/eldar/udl.txt")
        
     
        # iter_mc shape: (num_mc_simulations,)
        iter_mc = tf.keras.backend.variable(np.ones(num_mc_simulations))
        dist = tf.contrib.distributions.Normal(loc=tf.keras.backend.zeros_like(std), scale=std)
        mc_results = tf.keras.backend.map_fn(gaussian_categorical_crossentropy(true_labels, pred, dist, undistorted_loss, num_classes), iter_mc)
        variance_loss = tf.keras.backend.mean(mc_results, axis=0) * undistorted_loss
        #print_op = tf.print(variance_loss, output_stream=sys.stdout)
        variance_loss = tf.Print(variance_loss, [variance_loss], "\nMC_Distorted_loss: ", summarize=8)
        return variance_loss + undistorted_loss + variance_depressor
    return b_internal
        
def gaussian_categorical_crossentropy(true_labels, pred, dist, undistorted_loss, num_classes):
    def map_fn(i):
        std_samples = tf.keras.backend.transpose(dist.sample(num_classes))
        distorted_pred = pred + std_samples
        #distorted_loss = tf.keras.losses.categorical_crossentropy(true_labels, pred + std_samples, from_logits=True)
        distorted_loss = tf.keras.backend.categorical_crossentropy(true_labels, pred + std_samples, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -tf.keras.activations.elu(diff)
    return map_fn 
        
######################################################################




### Create Epistemic Uncertainty Model ###############################

def set_dropout(input_tensor, p=0.5, mc=True):
    if mc:
        seed = 1234
        tf.set_random_seed(seed)
        return tf.keras.layers.Dropout(rate=p, seed=seed)(input_tensor, training=True)
    else:
        return tf.keras.layers.Dropout(p)(input_tensor)


def create_epistemic_unc_model(num_features, num_classes, mc=True):
    print("\nEntering in create_epistemic_unc_model ...\n")
    input_layer = tf.keras.layers.Input(shape=(num_features, 1))
    print("\ninput_layer.shape: " + str(input_layer.shape))
    conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(input_layer)
    print("\nconv_1.shape: " + str(conv_1.shape))
    avgpool_a = tf.keras.layers.MaxPooling1D(pool_size=(8))(conv_1)
    dropout_a = set_dropout(avgpool_a, p=0.5, mc=mc)
    #dropout_a = tf.keras.layers.Dropout(0.5)(avgpool_a, training=True)
    conv_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=(5), padding='same', activation=tf.nn.relu)(dropout_a)
    dropout_b = set_dropout(conv_2, p=0.5, mc=mc)
    #dropout_b = tf.keras.layers.Dropout(0.5)(conv_2, training=True)
    flatten = tf.keras.layers.Flatten()(dropout_b)
    dense = tf.keras.layers.Dense(128, activation='relu')(flatten)
    dropout_c = set_dropout(dense, p=0.5, mc=mc)
    #dropout_c = tf.keras.layers.Dropout(0.5)(dense, training=True)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout_c)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

######################################################################


### Serving Input Function ###########################################

def s_input_fn(params, keras_input_names_list):
    image_tensor = tf.placeholder(tf.float32, [None, params.mfcc_size, 1])
    inputs = {keras_input_names_list: image_tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
    """
    inputs = {'b64': tf.placeholder(tf.string, shape=(None), name="encoded_image_string_tensor")}
    image_scalar = tf.squeeze(inputs['b64'])
    decoded_image = tf.to_float(tf.image.decode_image(image_scalar, params.num_channels))
    batched_image = tf.expand_dims(decoded_image, 0)
    image_tensor = tf.placeholder_with_default(batched_image, shape=[None, params.image_size, params.image_size, params.num_channels])
    image_tensor = image_tensor / 255.0
    features = {keras_input_names_list : image_tensor}
    return tf.estimator.export.ServingInputReceiver(features, inputs)
    """
    

def serving_input_fn(params, keras_input_names_list):
    return lambda: s_input_fn(params, keras_input_names_list)
