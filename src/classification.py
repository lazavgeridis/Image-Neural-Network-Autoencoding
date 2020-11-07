from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from os.path import exists
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", help="path to file containing the training set samples")
parser.add_argument("-dl", help="path to file containing the training set labels")
parser.add_argument("-t", help="path to file containing the test set samples")
parser.add_argument("-tl", help="path to file containing the test set labels")
parser.add_argument("-model", help="path to autoencoder model")
args = parser.parse_args()

# check if the file arguments exist
if not exists(args.d):
    die("\nFile \"{}\" does not exist!\n".format(args.d), -1)

if not exists(args.dl):
    die("\nFile \"{}\" does not exist!\n".format(args.dl), -1)

if not exists(args.t):
    die("\nFile \"{}\" does not exist!\n".format(args.t), -1)

if not exists(args.tl):
    die("\nFile \"{}\" does not exist!\n".format(args.tl), -1)

#if not exists(args.model):
#    die("\nFile \"{}\" does not exist!\n".format(args.model), -1)


# read files and store the data
trainset = dataset_reader(args.d)
_, x_dim, y_dim = np.shape(trainset)
trainlabels = labels_reader(args.dl)
testset = dataset_reader(args.t)
testset_size, _, _ = np.shape(testset)
testlabels = labels_reader(args.tl)

# plt.imshow(trainset[0, :, :], cmap='gray')
# plt.show()

# rescale pixels in [0, 1] and reshape
trainset = trainset / 255.0
testset  = testset / 255.0
trainset = trainset.reshape(-1, x_dim, y_dim, 1)
testset  = testset.reshape(-1, x_dim, y_dim, 1)
trainlabels = to_categorical(trainlabels, num_classes=10) # converts each class to one-hot encoding vector i.e 4 -> [0 0 0 1 0 0 0 0 0 0]

# CNN construction

""" 1) Input Layer: x_dim * y_dim dimensional data
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

cnn.add(Conv2D(32, kernel_size=5, activation="relu", input_shape=(x_dim, y_dim, 1), name="conv1"))
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

print("Number of layers is {}".format(len(cnn.layers)))
for l in cnn.layers:
    if l.name != "fully_connected": # 1st training stage: train only the weights of the fc layer, "freeze" the rest
        l.trainable = False

# retrieve the encoder (weights) from the saved autoencoder model
#cnn.load_weights(args.model, by_name=True)

# compile with Root Mean Square optimizer and Cross Entropy Cost
#cnn.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]) # optimizer="adam"


# train cnn
#option = -1
#while option == 1:
#    ep    = int(input("> Enter training epochs: "))
#    batch = int(input("> Enter training batch size: "))
#
#    cnn.fit(trainset, trainlabels, batch_size=batch, epochs=ep)
#    for l in cnn.layers:
#       l.trainable = True # 2nd training stage: train the entire network
#
#    re-compile the cnn and repeat the training
#    cnn.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
#    history = cnn.fit(trainset, trainlabels, batch_size=batch, epochs=ep)
#
#    print("""Training was completed. You now have the following options:
#            1. Repeat the training process with different number of epochs and/or batch size
#            2. Plot the accuracy and loss graphs with respect to the hyperparameters chosen and other evaluation metrics
#            3. Classify test set images (using other hyperparams????)
#            """)
#    option = int(input("> Enter the corresponding number: "))
#    if option < 1 or option > 3:
#        die("\nInvalid option!\n", -2)
#
## plot accuracy, loss
#if option == 2:
#    # model.evaluate() ????
#
## classification
#else:
#    y_pred = cnn.predict(testset) # it also has a batch_size argument
#    
#    cnt = 0
#    for i in testset_size:
#        if y_pred[i] == testlabels[i]:
#            cnt += 1
#    print("CNN Test Accuracy = %f" %(cnt/len(result)))
