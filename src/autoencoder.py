import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import dataset_reader
from keras.layers import layers
from sklearn.model_selection import train_test_split



class Autoencoder():

    def __init__(self, dataset, dims, epochs, batch_size, n_filters, filter_size, n_layers):
        self.dataset = dataset
        self.rows = dims[0]
        self.cols = dims[1]

        # Hyperparemeters
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_layers = n_layers

        self.autoencoder = None


    def split_dataset(self, testSize, randomState):
        # training data must be split into a training set and a validation set
        (trainSet, _), (testSet, _) = train_test_split(self.dataset, self.dataset, test_size=testSize, random_state=randomState)
        self.trainSet = trainSet
        self.testSet = testSet


    def reshape(self):
        
        if ((self.trainSet == None) or (self.testSet == None)):
            # we have to assing trainSet and testSet first in order to do the reshaping
            # apply split_dataset method before reshaping otherwise trainSet and testSet have "None" value
            # TODO::error msg and exiting
            pass

        x_train = self.trainSet.astype('float32') / 255.
        x_test = self.testSet.astype('float32') / 255.

        self.trainSet = np.reshape(x_train, (len(x_train), self.rows, self.cols, 1))
        self.testSet = np.reshape(x_test, (len(x_test), self.rows, self.cols, 1))


    def encoder(self, input_img):
        filters = self.n_filters
        
        conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same', name="conv1")(input_img)
        conv = layers.BatchNormalization(name="batch1")(conv)
        conv = layers.MaxPooling2D(pool_size=(3,3), padding="same", name="pool1")(conv)

        for i in range(1, self.n_layers):
            filters *= 2
            conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same', name="conv"+str(i+1))(conv)
            conv = layers.BatchNormalization(name="batch"+str(i+1))(conv)
            conv = layers.MaxPooling2D(pool_size=(3,3), padding="same", name="pool"+str(i+1))(conv)


        return conv


    def decoder(self):
        pass


    def compile_model(self):
        pass


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
    dataset, _, rows, cols = dataset_reader(dataset)
    dims = (rows, cols)

    testSize = 0.2
    randomState = 13

    autoencoder = Autoencoder(dataset, dims)
    autoencoder.split_dataset(testSize, randomState)
    autoencoder.reshape()

    input_img = keras.Input(shape(rows, cols, 1))
    encoded_img = autoencoder.encoder(input_img)

