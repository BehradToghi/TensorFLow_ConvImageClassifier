import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def tf_data_loader():
    from tensorflow import keras
    print("Reading Fashion MNIST dataset using Keras")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print("Dataset contains", train_images.shape[0], ' training and ', test_images.shape[0], ' test images of size: ',
          train_images.shape[1:3])
    return train_images, train_labels, test_images, test_labels


def plot_overall_stats(history, args):
    n_epochs = args.n_epochs
    scenario_name = args.scenario_name
    n_batch = args.n_batch
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(range(n_epochs), history['acc'], label='Training')
    ax1.plot(range(n_epochs), history['val_acc'], dashes=[6, 3], label='Validation')
    ax1.set(xlabel='Epoch', ylabel='Accuracy', title='Training and Validation Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.grid()

    ax2.plot(range(n_epochs), history['loss'], label='Training')
    ax2.plot(range(n_epochs), history['val_loss'], dashes=[6, 3], label='Validation')
    ax2.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
    ax2.legend(loc='upper right')
    ax2.grid()

    plt.suptitle(scenario_name + ', Epochs = ' + str(n_epochs) + ', Batch size = ' + str(n_batch))

    dir_path = os.path.join('outputs', 'plots')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    temp_path = os.path.join(dir_path, scenario_name)

    plt.savefig(temp_path + '.png', bbox_inches='tight')

    return


def save_object(obj_list, pkl_name):
    dir_path = os.path.join('outputs', 'pickles')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pkl_path = os.path.join(dir_path, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(obj_list, f)

    return


def load_object(file_path):
    with open(file_path, 'rb') as f:
        readout = pickle.load(f)

    return readout

