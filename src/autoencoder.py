import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import Conv2D, MaxiPooling2D
from sklearn.model_selection import train_test_split
from keras.models import load_model



if __name__ == '__main__':

    # parsing the command line arguments with argparse module
    # --help provides instructions
    parser = argparse.ArgumentParser(description = 'Provide dataset file')
    parser.add_argument('-d', '--dataset', required = True, help = 'dataset file path')
    args = vars(parser.parse_args())

    # if dataset file exists, otherwise exit
    dataset = args['dataset']
    if (os.path.isfile(dataset) == False):
        print("\n[+] Error: This dataset file does not exist\n\nExiting..")
        sys.exit(-1)

    # read the first 16 bytes (magic_number, number_of_images, rows, columns)
    with open(dataset, "rb") as f:
        train = f.read()
        images_n = int.from_bytes(train[4:8], byteorder='big')
        print("Number of images to train {}". format(images_n))
        rows = int.from_bytes(train[8:12], byteorder='big')
        print("Number of rows: {}". format(rows))
        cols = int.from_bytes(train[12:16], byteorder='big')
        print("Number of columns: {}". format(cols))

        # create np.ndarray for every 784 (rows * cols) pixels (maybe well change that)
        # append each np.ndarray in a list
        # efficient way to read a binary file and convert it into np.ndarray
        dt = np.dtype(np.uint8)
        offset = 16
        trainImages = []
        for i in range(images_n):
            img = np.frombuffer(train[offset:(rows*cols)+offset], dt)
            # reshape image into 2d np.array
            img = np.reshape(img, (-1, cols))
            trainImages.append(img)
            offset += rows * cols


    """
    We'll discuss these shits
    Hyperparameters:
    number_of_filters
    filter_size
    number_of_layers
    number_of_epochs
    batch_size
    learning_rate
    """

    # training data must be split into a training set and a validation set
    train_x, valid_x, train_ground, valid_ground = train_test_split(trainImages, trainImages, test_size=0.2, random_state=13)
    if (train_x[0].all == train_ground[0].all):
        print("same")

    # plot first image (just to check if everything's correct)
    # plt.imshow(train_x[0])
    # plt.show()
    # after plotting everything's is perfect!

    # we have to convert nparray to tensor
    # tensor = tf.convert_to_tensor(train_x[0], dtype=tf.float32)
    # then we have to reshape it in 4dims (-1, rows, cols, initial_number_of_filters (1))
    # arg = tf.reshape(tensor, [-1, rows, cols, 1])
    # apply a conv2d layer
    # conv1 = Conv2D(32, kernel_size = 3, activation='relu', padding='same')(arg)
    # not sure if this is correct!!
    # print(conv1)