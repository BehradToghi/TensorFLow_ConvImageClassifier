import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from utils.mnist_utils import tf_data_loader
from utils.mnist_utils import plot_overall_stats
import utils as utils


def main(args):
    print("Using TensorFlow V.", tf.__version__)

    # Initialization
    n_epochs = args.n_epochs
    scenario_name = args.scenario_name
    n_batch = args.n_batch

    # Read data
    train_images, train_labels, test_images, test_labels = tf_data_loader()
    # Normalize data
    train_images = train_images/255
    test_images = test_images/255

    # Build the model
    if args.mode == 'FC':
        print("Experiment With Fully-connected network")
        model = tf.keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))
        model.add(keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(keras.layers.Dense(10, activation=tf.keras.activations.softmax))

    elif args.mode == 'Conv':
        print("Experiment With Convolutional network")
        model = tf.keras.Sequential()
        model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation=tf.keras.activations.relu, input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.keras.activations.relu))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation=tf.keras.activations.softmax))

        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    else:
        raise Exception('Please enter --mode=FC/Conv')

    # Compile the model with desired optimizer and loss function
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    History = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=n_batch,
        epochs=n_epochs,
        verbose=1,
        callbacks=None,
        validation_split=0.10
    )

    model.summary()

    # Some practice with methods of Keras layers
    for layer in model.layers:
        print('***layer name:    ', layer.get_config()['name'])
        print('***.input_shape:    ', layer.input_shape)
        print('***.output_shape:    ', layer.output_shape)

    # Save the training history
    utils.mnist_utils.save_object(History.history, scenario_name + '.pkl')
    training_history = utils.mnist_utils.load_object(os.path.join('outputs', 'pickles', scenario_name + '.pkl'))

    # Plot training stats
    plot_overall_stats(training_history, args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple Fashion MNIST Classifier')
    parser.add_argument('--mode', type=str, default='FC', help='Architecture of the network, FC or ConvNet.')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of Epochs for Training.')
    parser.add_argument('--n_batch', type=int, default=32, help='Size of the mini-batch.')
    parser.add_argument('--scenario_name', type=str, default='Training',
                        help='Name of the scenario for saving output files.')
    args = parser.parse_args()
    main(args)
