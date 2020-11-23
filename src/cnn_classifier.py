from keras.models import Sequential, load_model
from keras.layers import MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import RMSprop, Adam
from keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np


# default values
LEARNINGRATE = 2e-4
DROPOUT      = 0.4  # 0.5


class Classifier:

    def __init__(self, ae_model):
        self.model = Sequential()
        self.train_history = None

        # each layer name in the autoencoder ends with either 'e' or 'd'
        # 'e': encoder layer, 'd': decoder layer
        for i in range(len(ae_model.layers)):
          if ae_model.layers[i].name[-1] == "d":
              break
          self.model.add(ae_model.get_layer(index=i))

          # after each max pooling layer add a dropout layer
          if ae_model.layers[i].name[:4] == "pool":
              self.model.add(Dropout(DROPOUT))

        self.model.add(Flatten())


    def add_layers(self, fc_nodes):
        self.model.add(Dense(fc_nodes, activation="relu", name="fully_connected"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(10, activation="softmax", name="output"))
        print(self.model.summary())


    def train(self, train_images, train_labels, val_images, val_labels, batchsize, num_epochs):
        for l in self.model.layers:
            if l.name != "fully_connected": # 1st training stage: train only the weights of the fc layer, "freeze" the rest
                l.trainable = False

        # compile 
        self.model.compile(loss=SparseCategoricalCrossentropy(),
                            optimizer=Adam(learning_rate=LEARNINGRATE),
                            metrics=[SparseCategoricalAccuracy()])       

        print("\nTraining Stage 1: Training only the Fully-Connected layer's weights...")
        self.model.fit(train_images, train_labels, batch_size=batchsize, epochs=num_epochs, validation_data=(val_images, val_labels))
        print("Done!\n")

        for l in self.model.layers:
            l.trainable = True # 2nd training stage: train the entire network

        # re-compile the cnn and repeat the training
        self.model.compile(loss=SparseCategoricalCrossentropy(),
                            optimizer=Adam(learning_rate=LEARNINGRATE),
                            metrics=[SparseCategoricalAccuracy()])       

        print("\nTraining Stage 2: Training the entire network...")
        self.train_history = self.model.fit(train_images, train_labels, batch_size=batchsize, 
                                        epochs=num_epochs, validation_data=(val_images, val_labels))
        print("Done!\n")


    def test(self, test_images, test_labels, size):
        y_pred1 = self.model.predict(test_images)
        y_pred2 = np.argmax(y_pred1, axis=1)

        cnt = 0
        for i in range(size):
            if y_pred2[i] == test_labels[i]:
                cnt += 1
        print("\nClassifier Test Accuracy = {}".format(cnt / len(y_pred2)))

        return (y_pred2, cnt / len(y_pred2) )
