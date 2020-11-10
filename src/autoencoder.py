import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import keras

from utils import dataset_reader, die, ask_for_hyperparameters
from cnn_model import Autoencoder


def plot_reconstructed_digits(autoencoder, x_test):
    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



if __name__ == '__main__':

    # parsing the command line arguments with argparse module
    # --help provides instructions
    parser = argparse.ArgumentParser(description = 'Provide dataset file')
    parser.add_argument('-d', '--dataset', required = True, help = 'dataset file path')
    args = vars(parser.parse_args())

    # if dataset file exists, otherwise exit
    dataset = args['dataset']
    # dataset = 'train-images-idx3-ubyte'
    if (os.path.isfile(dataset) == False):
        print("\n[+] Error: This dataset file does not exist\n\nExiting..")
        sys.exit(-1)


    # read the first 16 bytes (magic_number, number_of_images, rows, columns)
    dataset, _, rows, cols = dataset_reader(dataset)
    dims = (rows, cols)

    testSize = 0.2
    epochs, batch_size, convs = ask_for_hyperparameters()

    autoencoder = Autoencoder(dataset, dims, epochs, batch_size, convs)
    autoencoder.split_dataset(testSize)
    autoencoder.reshape()

    input_img = keras.Input(shape=(rows, cols, 1))
    encoded = autoencoder.encoder(input_img)
    decoded = autoencoder.decoder(encoded)
    autoencoder.compile_model(input_img, decoded)
    autoencoder.train_model()
    # close pop-up window after plotting in order to continue
    # plot_reconstructed_digits(autoencoder.autoencoder, autoencoder.testSet)
    path = input("> Give the path where the CNN will be saved: ")
    autoencoder.save_model(path)

