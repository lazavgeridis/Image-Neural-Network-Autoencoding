import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import keras

from utils import dataset_reader, die, ask_for_hyperparameters
from cnn_model import Autoencoder


def cnn_train_simulation():
    epochs, batch_size, convs = ask_for_hyperparameters()

    autoencoder = Autoencoder(dataset, dims, epochs, batch_size, convs)
    autoencoder.split_dataset(testSize)
    autoencoder.reshape()

    input_img = keras.Input(shape=(rows, cols, 1))
    encoded = autoencoder.encoder(input_img)
    decoded = autoencoder.decoder(encoded)
    autoencoder.compile_model(input_img, decoded)
    autoencoder.train_model()

    return autoencoder


def menu():
    print("\n1. Do you want to repeat the experiment with other hyperparameter values?")
    print("2. Do you want the error graphs to be displayed as to the values ​​of the hyperparameters for the performed experiments?")
    print("3. Do you want to save the trained model with the latest hyperparameter values?")
    print("4. Do you want to exit the program?")
    code = input("Provide one of the above codes: ")
    
    return code



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

    autoencoder = cnn_train_simulation()

    while (True):
        code = menu()
        if (code == '1'):
            autoencoder = cnn_train_simulation()
        if (code == '2'):
            # autoencoder.plot_loss()
            pass
        elif (code == '3'):
            path = input("> Give the path where the CNN will be saved: ")
            autoencoder.save_model(path)
        elif (code == '4'):
            sys.exit(0)
        else:
            print("[+] Error: Provide only one of the above codes (1, 2, 3, 4)")
            continue

