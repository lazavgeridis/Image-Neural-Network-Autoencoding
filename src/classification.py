from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from os.path import exists
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-d", help="path to file containing the training set samples")
parser.add_argument("-dl", help="path to file containing the training set labels")
parser.add_argument("-t", help="path to file containing the test set samples")
parser.add_argument("-tl", help="path to file containing the test set labels")
parser.add_argument("-model", help="path to autoencoder model")
args = parser.parse_args()

# check if the file arguments exist
if not exists(args.d):
    print("\nFile {} does not exist!\n".format(args.d), file=sys.stderr)
    sys.exit(-1)

if not exists(args.dl):
    print("\nFile {} does not exist!\n".format(args.dl), file=sys.stderr)
    sys.exit(-1)

if not exists(args.t):
    print("\nFile {} does not exist!\n".format(args.t), file=sys.stderr)
    sys.exit(-1)

if not exists(args.tl):
    print("\nFile {} does not exist!\n".format(args.tl), file=sys.stderr)
    sys.exit(-1)

#if not exists(args.model):
#    print("\nFile {} does not exist!\n".format(args.model), file=sys.stderr)
#    sys.exit(-1)


# read the file arguments




""" 1) Input Layer: 784 dimensional data
    2) Layer conv1 has 32 feature maps, 5x5 each, stride = 1, TEST padding = 'valid' (default) / 'same' 
    3) Batch normalization, followed by Max pooling 2x2, followed by Dropout 
    4) Layer conv2 has 64 feature maps, 5x5 each, stride = 1, TEST padding = 'valid' (default) / 'same'
    5) Batch normalization, followed by Max pooling 2x2, followed by Dropout 
    6) Fully Connected layer with _ nodes
    7) Dropout
    8) Output layer with softmax
"""
   
dropout  = 0.4 # 0.5
fc_nodes = 128
cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=5, activation="relu", input_shape=(28, 28, 1), name="conv1"))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D())
cnn.add(Dropout(dropout))

cnn.add(Conv2D(64, kernel_size=5, activation="relu", name="conv2"))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D())
cnn.add(Dropout(dropout))

cnn.add(Flatten())
cnn.add(Dense(fc_nodes, activation="relu", name="fully_connected"))
#cnn.add(BatchNormalization())
cnn.add(Dropout(dropout))
cnn.add(Dense(10, activation="softmax", name="output"))

# retrieve the encoder (weights) from the saved autoencoder model
#cnn.load_weights(args.model, by_name=True)
#model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]) # optimizer="adam"
