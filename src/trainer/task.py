"""Train the model"""

import argparse
import model
import os
import sys
import tensorflow as tf
import utils
from tensorflow.python.keras._impl import keras
from tensorflow.python.lib.io import file_io




parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='../experiments/base_model/',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='../data/',
                    help="Directory containing the dataset (Train and Test TFRecords)")
parser.add_argument('--job-dir', default='../experiments/base_model/',
                    help="GCloud parameter")
parser.add_argument('--export-graph', action='store_true', help="Only exports model graph")
parser.add_argument('--export-dir', default='../experiments/base_model/export/exporter/',
                    help="Directory where the model will be exported")
parser.add_argument('--mode', default='train', help="mode= train / test")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load the parameters from json file
    args = parser.parse_args()
    train_path = os.path.join(args.data_dir, 'train.record')
    test_path = os.path.join(args.data_dir, 'test.record')
    json_path = os.path.join(args.config_dir, 'params.json')
    mode = args.mode

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
    

    # Creating the model using Keras Functional API (params.mfcc_size = 40, params.num_classes = 5)
    cnn_model = model.conv_model(params.mfcc_size, params.num_classes)

    # The used optimizer in the original configuration was RMSProp with the following params: 
    # opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    keras_input_names_list = 'input_1'
    estimator = tf.keras.estimator.model_to_estimator(keras_model=cnn_model, model_dir=args.job_dir, config=config)
    if args.export_graph:
        print(params)
        print("\nExporting the graph to " + args.export_dir + "...")
        estimator.export_savedmodel(args.export_dir, model.serving_input_fn(params, keras_input_names_list), strip_default_attrs=True)
        sys.exit(0)

    num_examples = 0
    for record in tf.python_io.tf_record_iterator(train_path):
        num_examples += 1
    nbatches_per_epoch = int(num_examples / params.batch_size)
    max_steps_train = params.num_epochs * nbatches_per_epoch
    print("\nMAX STEPS TRAIN " + str(max_steps_train))

    # Train and evaluate the model
    tf.logging.info("\nStarting training for {} epoch(s).".format(params.num_epochs))
    train_spec = tf.estimator.TrainSpec(input_fn=model.train_input_fn(train_path, params, num_examples, keras_input_names_list), max_steps=None)
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn(params, keras_input_names_list), exports_to_keep=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=model.test_input_fn(test_path, params, keras_input_names_list), steps=100, exporters=exporter)
    #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if mode == 'train':
        estimator.train(input_fn=model.train_input_fn(train_path, params, num_examples, keras_input_names_list), steps=100000)
    elif mode == 'test':
        result = estimator.evaluate(input_fn=model.test_input_fn(test_path, params, keras_input_names_list), steps=100)
        print(result)

    """
    estimator.train(lambda: train_input_fn(args.data_dir, params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
    """
