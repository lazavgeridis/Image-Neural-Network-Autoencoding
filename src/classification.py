from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from os.path import exists
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

# default values
FC_NODES = 128
DROPOUT  = 0.4 # 0.5


def main():
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
    
    if not exists(args.model):
        die("\nFile \"{}\" does not exist!\n".format(args.model), -1)
    
    
    # read files and store the data
    trainset, _, x_dim, y_dim = dataset_reader(args.d)
    trainlabels = labels_reader(args.dl)
    testset, testset_size, _, _ = dataset_reader(args.t)
    testlabels = labels_reader(args.tl)
    
    # plt.imshow(trainset[0, :, :], cmap='gray')
    # plt.show()
    
    # rescale pixels in [0, 1]
    trainset = trainset / 255.0
    testset  = testset / 255.0

    # reshape the data, tensors of shape (size, rows, cols, inchannel)
    trainset = trainset.reshape(-1, x_dim, y_dim, 1)
    testset  = testset.reshape(-1, x_dim, y_dim, 1)

    # reserve 0.2*training samples for validation
    x_train, x_val, y_train, y_val = train_test_split(trainset, trainlabels, test_size=0.2)


    # CNN Classifier Construction

    """ 6) Fully Connected layer with _ nodes
        7) Dropout
        8) Output layer with softmax
    """
       
    ae = load_model(args.model)
    print("Number of layers in autoencoder is {}".format(len(ae.layers)))
    for l in ae.layers:
        print(l.name, l.trainable)

    cnn_classifier = Sequential()

    for i in range(0, len(ae.layers)):
        if ae.layers[index].name == "conv1d":
            break
        cnn_classifier.add(ae.get_layer(index=i))
        # after each max pooling layer add a dropout layer
        if ae.layers[ind].name[:4] == "pool":
            cnn_classifier.add(Dropout(DROPOUT))
        
    # Now our cnn classifier stores only the encoder architecture
    # Add: Flatten - Dense / Fully Connected - Dropout - Output (softmax)
    cnn_classifier.add(Flatten())
    cnn_classifier.add(Dense(FC_NODES, activation="relu", name="fully_connected"))
    #cnn.add(BatchNormalization())
    cnn_classifier.add(Dropout(DROPOUT))
    
    cnn_classifier.add(Dense(10, activation="softmax", name="output"))

    print(cnn_classifier.summary())
    
    for l in cnn_classifier.layers:
        if l.name != "fully_connected": # 1st training stage: train only the weights of the fc layer, "freeze" the rest
            l.trainable = False
    
    # compile with Root Mean Square optimizer and Cross Entropy Cost
    cnn_classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
                            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), 
                            metrics=[keras.metrics.SparseCategoricalAccuracy()])       # optimizer=keras.optimizers.Adam()
    
    # Train CNN 
    option = 1
    while option == 1:
        ep    = int(input("> Enter training epochs: "))
        batch = int(input("> Enter training batch size: "))
    
        print("\nTraining Stage 1: Training only the Fully-Connected layer's weights...")
        history1 = cnn_classifier.fit(x_train, y_train, batch_size=batch, epochs=ep, validation_data=(x_val, y_val))
        print("Done!\n")

        for l in cnn_classifier.layers:
           l.trainable = True # 2nd training stage: train the entire network
    
        # re-compile the cnn and repeat the training
        cnn_classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
                                optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), 
                                metrics=[keras.metrics.])       # optimizer=keras.optimizers.Adam()

        print("\nTraining Stage 2: Training the entire network...")
        history2 = cnn_classifier.fit(x_train, y_train, batch_size=batch, epochs=ep, validation_data=(x_val, y_val))
        print("Done!\n")
    
        print("""\nTraining was completed. You now have the following options:
                1. Repeat the training process with different number of epochs and/or batch size
                2. Plot the accuracy and loss graphs with respect to the hyperparameters chosen and other evaluation metrics
                3. Classify test set images (using other hyperparams????)
                """)
        
        option = int(input("> Enter the corresponding number: "))
        if option < 1 or option > 3:
            die("\nInvalid option!\n", -2)
    
    # plot accuracy, loss
    if option == 2:
        results = cnn_classifier.evaluate(testset, testlabels, batch_size=batch)
        print("Test Loss, Test Accuracy:", results)
    
    # classification
    else:
        y_pred = cnn_classifier.predict(testset) # it also has a batch_size argument
        
        cnt = 0
        for i in testset_size:
            if y_pred[i] == testlabels[i]:
                cnt += 1
        print("CNN Test Accuracy = %f" %(cnt/len(y_pred)))


if __name__ == '__main__':
    main()
