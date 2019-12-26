"""Train the model

Train using epistemic model:
python task.py --job-dir ../experiments/ebayes_keras --config_dir ../experiments/ebayes_keras/ --bayesian_unc epistemic --latest_ckpt ../experiments/ebayes_keras/cp-0100.ckpt

Train using aleatoric keras model:
python task.py --job-dir ../experiments/abayes_keras/ --config_dir ../experiments/abayes_keras/ --bayesian_unc al 
eatoric

Export epistemic model:
python task.py --export-graph --export-dir ../experiments/ebayes_keras/export/exporter/1/

"""
from __future__ import division, absolute_import, print_function
from builtins import *
import argparse
import model
import os
import sys
import tensorflow as tf
import utils
#from tensorflow.python.keras._impl import keras
from tensorflow.python.lib.io import file_io




parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='../experiments/base_model/',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='../data/',
                    help="Directory containing the dataset (Train and Test TFRecords)")
parser.add_argument('--job-dir', default='../experiments/base_model/',
                    help="Local or GCloud path to store checkpoints")
parser.add_argument('--export-graph', action='store_true', help="Only exports model graph")
parser.add_argument('--export-dir', default='../experiments/base_model/export/exporter/1/',
                    help="Directory where the model will be exported")
parser.add_argument('--mode', default='train', help="mode= train / test")
parser.add_argument('--bayesian_unc', default='simple', 
                    help='bayesian uncertainty model. Possible values: aleatoric | epistemic | simple')
parser.add_argument('--latest_ckpt', help='Path of the latest checkpoint')



if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load the parameters from json file
    args = parser.parse_args()
    train_path = os.path.join(args.data_dir, 'train.record')
    test_path = os.path.join(args.data_dir, 'test.record')
    json_path = os.path.join(args.config_dir, 'params.json')
    mode = args.mode
    bayes_unc = args.bayesian_unc

    try:
        file_io.stat(json_path)
    except Exception:
        print("\nNo json configuration file found at {}".format(json_path))
        sys.exit(-1)
    #assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # Define the model
    tf.logging.info("\nCreating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    save_checkpoints_steps=10000, 
                                    save_checkpoints_secs=None,
                                    keep_checkpoint_max = 3,
                                    save_summary_steps=params.save_summary_steps)
    
    if bayes_unc == 'aleatoric':
        print("\nCreating Bayesian Aleatoric Model ...\n")
        cnn_model = model.create_bayesian_model(params.mfcc_size, params.num_classes)
        print("\nCompiling Bayesian Aleatoric Model ...\n")
        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss={'logits_variance': model.bayesian_categorical_crossentropy(params.num_mc_simulations, params.num_classes), 'softmax_output': tf.keras.losses.categorical_crossentropy}, metrics={'softmax_output': 'accuracy'}, loss_weights={'logits_variance': 0.3, 'softmax_output': 0.7})

    else:
        # Creating the model using Keras Functional API (params.mfcc_size = 40, params.num_classes = 5)
        cnn_model = model.conv_model(params.mfcc_size, params.num_classes)        
        # The used optimizer in the original configuration was RMSProp with the following params: 
        # opt = tf.keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    keras_input_names_list = 'input_1'
    checkpoint_path = os.path.join(args.job_dir, "cp-{epoch:05d}.ckpt")
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    #files_ckpt = sorted(glob.glob(os.path.join(chekpoint_dir, 'cp-????.ckpt')))
    print("\nlatest_checkpoint: ")
    print(latest)
  
    if latest != None:
        print("\nLoading weights ...")
        try:
            cnn_model.load_weights(latest)
            initial_epoch = int(latest[-9:-5])
            print("\nInitial epoch: " + str(initial_epoch))
            print("\nModel loaded")
        except:
            print("\nThe model hasn't loaded the weights")
    else:
        initial_epoch = 0
    """
    print("\nCreating Estimator from Keras Model ...\n")
    estimator = tf.keras.estimator.model_to_estimator(keras_model=cnn_model, model_dir=args.job_dir, config=config)

    if args.export_graph:
        print(params)
        print("\nExporting the graph to " + args.export_dir + "...\n")
        estimator.export_savedmodel(args.export_dir, model.serving_input_fn(params, keras_input_names_list), strip_default_attrs=True)
        sys.exit(0)
    """

    if args.export_graph:
        print(params)
        print("\nExporting the graph to " + args.export_dir + "...\n")
        # '../experiments/base_model/export/exporter/1/' where the last path component (1/ here) is a version number for your model - it allows tools like Tensorflow Serving to reason about the relative freshness.
        tf.saved_model.save(cnn_model, args.export_dir)
        sys.exit(0)


    num_examples = 0
    for record in tf.python_io.tf_record_iterator(train_path):
        num_examples += 1
    print("\nNUM TRAINING EXAMPLES: " + str(num_examples))
    print("\nBATCH SIZE: " + str(params.batch_size))
    nbatches_per_epoch = int(num_examples / params.batch_size)
    print("\nNUM STEPS PER EPOCH: " + str(nbatches_per_epoch))
    max_steps_train = params.num_epochs * nbatches_per_epoch
    print("\nMAX STEPS TRAIN " + str(max_steps_train) + "\n")



    # Create a callback that saves the model's weights every 1 epoch
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(args.job_dir, "cp-{epoch:04d}.ckpt") 
    #checkpoint_path = args.job_dir + "cp.ckpt"
    #checkpoint_path = args.job_dir + "cp-{epoch:04d}.hdf5" 
    chkp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=10)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)

    # Train and evaluate the model
    tf.logging.info("\nStarting training for {} epoch(s).\n".format(params.num_epochs))
    trainx, (trainy_1, trainy_2) = model.input_fn(train_path, params, num_examples, True, keras_input_names_list, bayes_unc)
    testx, (testy_1, testy_2) = model.input_fn(test_path, params, 0, False, keras_input_names_list, bayes_unc)
    epochs = params.num_epochs + initial_epoch
    cnn_model.fit(trainx, (trainy_1, trainy_2), epochs=epochs, steps_per_epoch=nbatches_per_epoch, validation_data=(testx, (testy_1, testy_2)), validation_steps=1, initial_epoch=initial_epoch, callbacks=[chkp_cb, reduce_lr])
    #cnn_model.fit(trainx, (trainy_1, trainy_2), epochs=params.num_epochs, steps_per_epoch=nbatches_per_epoch, callbacks=[chkp_cb])
    cnn_model.save(os.path.join(args.job_dir, 'my_model.h5'))




    """
    train_spec = tf.estimator.TrainSpec(input_fn=model.train_input_fn(train_path, params, num_examples, keras_input_names_list, bayes_unc), max_steps=None)
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn(params, keras_input_names_list), exports_to_keep=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=model.test_input_fn(test_path, params, keras_input_names_list, bayes_unc), steps=100, exporters=exporter)
    #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if mode == 'train':
        estimator.train(input_fn=model.train_input_fn(train_path, params, num_examples, keras_input_names_list, bayes_unc), steps=100000)
        
    elif (mode == 'test') and (bayes_unc == 'simple'):
        result = estimator.evaluate(input_fn=model.test_input_fn(test_path, params, keras_input_names_list, bayes_unc), steps=100)
        print("\nTest Loss and Accuracy of " + str(bayes_unc) + " model: ")
        print(result)

    """
    """
    estimator.train(lambda: train_input_fn(args.data_dir, params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
    """
