"""Define the model."""

import tensorflow as tf
import numpy  as np

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
    # Get the label associated with the image.
    label = parsed_example['label']
    label = tf.one_hot(label, depth=num_classes)
    # The image and label are now correct TensorFlow types.
    return audio, label

def input_fn(filename_tfrecords, params, num_examples, train, keras_input_names_list):
    global num_classes, mfcc_size
    num_classes = params.num_classes
    mfcc_size = params.mfcc_size
    dataset = tf.data.TFRecordDataset(filenames=filename_tfrecords)
    dataset = dataset.map(parse)
    if train:
        dataset = dataset.shuffle(num_examples) # whole dataset into the buffer
        num_repeat = params.num_epochs
    else:
        num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1) # make sure you always have one batch ready to serve
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    #x = {'image': images_batch}
    x = {keras_input_names_list: images_batch}
    y = labels_batch
    return x, y

def input_fn_pred(numpy_speech, params, keras_input_names_list):
    x_pred = tf.convert_to_tensor(numpy_speech, dtype=tf.float32)
    x_pred = tf.reshape(x_pred, (-1, params.mfcc_size, 1))
    x = {keras_input_names_list: x_pred}
    return x
    

def train_input_fn(path_tfrecords_train, params, num_examples, keras_input_names_list):
    return lambda: input_fn(path_tfrecords_train, params, num_examples, True, keras_input_names_list)


def test_input_fn(path_tfrecords_test, params, keras_input_names_list):
    return lambda: input_fn(path_tfrecords_test, params, 0, False, keras_input_names_list)

def pred_input_fn(numpy_speech, params, keras_input_names_list):
    # numpy_speech: the audio should be opened using ... numpy_speech, sr = librosa.load(path, res_type='kaiser_best', sr=sample_rate, mono=mono)
    return lambda: input_fn_pred(numpy_speech, params, keras_input_names_list)


### Simple CNN model using Keras Functional API ###

def conv_model(num_features, num_classes):
    input_layer = tf.keras.layers.Input(shape=(num_features, 1))
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


    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


########################################################


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
