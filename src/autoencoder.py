import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from utils import dataset_reader, die, ask_for_hyperparameters
from keras import layers
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split



class Autoencoder():

    def __init__(self, dataset, dims, epochs, batch_size, n_filters, filter_size):
        self.dataset = dataset
        self.rows = dims[0]
        self.cols = dims[1]

        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.filter_size = filter_size

        self.autoencoder = None


    def split_dataset(self, testSize, randomState=13):
        # training data must be split into a training set and a validation set
        self.trainSet, _, self.testSet, _ = train_test_split(self.dataset, self.dataset, test_size=testSize, random_state=randomState)


    def reshape(self):   
        if ((self.trainSet.any() == None) or (self.testSet.any() == None)):
            # we have to assing trainSet and testSet first in order to do the reshaping
            # apply split_dataset method before reshaping otherwise trainSet and testSet have "None" value
            die("\nApply split_dataset method before reshaping\nExiting..\n", -1)

        x_train = self.trainSet.astype('float32') / 255.
        x_test = self.testSet.astype('float32') / 255.

        self.trainSet = np.reshape(x_train, (len(x_train), self.rows, self.cols, 1))
        self.testSet = np.reshape(x_test, (len(x_test), self.rows, self.cols, 1))


    def encoder(self, input_img):
        filters = self.n_filters
        conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same', name='conv1')(input_img)
        conv = layers.BatchNormalization(name='batch1')(conv)
        conv = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='pool1')(conv)

        for i in range(1, 2):
            filters *= 2
            conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same', name='conv'+str(i+1))(conv)
            conv = layers.BatchNormalization(name='batch'+str(i+1))(conv)
            conv = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='pool'+str(i+1))(conv)

        filters *= 2
        conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same', name='conv3')(conv)
        encoded = layers.BatchNormalization(name='batch3')(conv)

        return (encoded, filters)


    def decoder(self, encoded, filters):
        conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same')(encoded)
        conv = layers.BatchNormalization()(conv)

        for _ in range(1, 2):
            filters /= 2
            conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same')(conv)
            conv = layers.BatchNormalization()(conv)
            conv = layers.UpSampling2D((2,2))(conv)

        filters /= 2
        conv = layers.Conv2D(filters, kernel_size=self.filter_size, activation='relu', padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.UpSampling2D((2,2))(conv)

        return layers.Conv2D(1, kernel_size=self.filter_size, activation='sigmoid', padding='same')(conv)


    def compile_model(self, input_img, decoded):
        self.autoencoder = keras.Model(input_img, decoded)
        print(self.autoencoder.summary())
        opt = keras.optimizers.RMSprop(learning_rate=0.1)
        self.autoencoder.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])


    def train_model(self):
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
        self.autoencoder.fit(self.trainSet, self.trainSet, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.testSet, self.testSet), callbacks=[annealer])


    def save_weights(self):
        self.autoencoder.save_weights('autoencoder.h5')


def reconstructed_digits(autoencoder, x_test):
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
    if (os.path.isfile(dataset) == False):
        print("\n[+] Error: This dataset file does not exist\n\nExiting..")
        sys.exit(-1)


    # read the first 16 bytes (magic_number, number_of_images, rows, columns)
    dataset, _, rows, cols = dataset_reader(dataset)
    dims = (rows, cols)

    testSize = 0.2
    n_filters = 32
    epochs, batch_size, filter_size = ask_for_hyperparameters()


    autoencoder = Autoencoder(dataset, dims, epochs, batch_size, n_filters, filter_size)
    autoencoder.split_dataset(testSize)
    autoencoder.reshape()

    input_img = keras.Input(shape=(rows, cols, 1))
    encoded, filters = autoencoder.encoder(input_img)
    decoded = autoencoder.decoder(encoded, filters)
    autoencoder.compile_model(input_img, decoded)
    autoencoder.train_model()
    reconstructed_digits(autoencoder.autoencoder, autoencoder.testSet)
    autoencoder.save_weights()

