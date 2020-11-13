from keras.models import Sequential, load_model
from keras.layers import MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import RMSprop, Adam
from keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


# default values
LEARNINGRATE = 1e-3
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

        self.model.add(Flatten())   # maybe not required?


    def add_layers(self, fc_nodes):
        self.model.add(Dense(fc_nodes, activation="relu", name="fully_connected"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(10, activation="softmax", name="output"))
        print(self.model.summary())


    def remove_layers():
        for i in range(4):
            self.model.pop()


    def train(self, train_images, train_labels, val_images, val_labels, batchsize, num_epochs):
        for l in self.model.layers:
            if l.name != "fully_connected": # 1st training stage: train only the weights of the fc layer, "freeze" the rest
                l.trainable = False

        # compile with Root Mean Square optimizer and Cross Entropy Cost
        self.model.compile(loss=SparseCategoricalCrossentropy(),
                            optimizer=RMSprop(learning_rate=LEARNINGRATE),
                            metrics=[SparseCategoricalAccuracy()])       # optimizer=keras.optimizers.Adam()

        print("\nTraining Stage 1: Training only the Fully-Connected layer's weights...")
        self.model.fit(train_images, train_labels, batch_size=batchsize, epochs=num_epochs, validation_data=(val_images, val_labels))
        print("Done!\n")

        for l in self.model.layers:
            l.trainable = True # 2nd training stage: train the entire network

        # re-compile the cnn and repeat the training
        self.model.compile(loss=SparseCategoricalCrossentropy(),
                            optimizer=RMSprop(learning_rate=LEARNINGRATE),
                            metrics=[SparseCategoricalAccuracy()])       # optimizer=keras.optimizers.Adam()

        print("\nTraining Stage 2: Training the entire network...")
        self.history = self.model.fit(train_images, train_labels, batch_size=batchsize, 
                                        epochs=num_epochs, validation_data=(val_images, val_labels))
        print("Done!\n")


    def test(self, test_images, test_labels, size):
        y_pred1 = self.model.predict(test_images)
        y_pred2 = np.argmax(y_pred1, axis=1)

        cnt = 0
        for i in range(size):
            if y_pred2[i] == test_labels[i]:
                cnt += 1

        print("\nClassifier Test Accuracy = {}".format( cnt/len(y_pred2) ))
        self.visualize_predictions(test_images, test_labels, y_pred2)


    def plot_train_curve(self, val_images, val_labels, batchsize, fc_nodes):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        fig.tight_layout()

        ax[0].plot(self.history.history["sparse_categorical_accuracy"])
        ax[0].plot(self.history.history["val_sparse_categorical_accuracy"])
        ax[0].set_title("Training Curve (batch_size={}, fc_nodes={})".format(batchsize, fc_nodes))
        ax[0].set_ylabel("accuracy")
        ax[0].set_xlabel("epochs")
        ax[0].legend(["training accuracy", "validation accuracy"], loc="lower right")

        ax[1].plot(self.history.history["loss"])
        ax[1].plot(self.history.history["val_loss"])
        ax[1].set_title("Training Curve (batch_size={}, fc_nodes={})".format(batchsize, fc_nodes))
        ax[1].set_ylabel("loss")
        ax[1].set_xlabel("epochs")
        ax[1].legend(["training loss", "validation loss"], loc="upper right")

        fig.subplots_adjust(hspace=1)
        plt.show()

        y_val_pred1 = self.model.predict(val_images)
        y_val_pred2 = np.argmax(y_val_pred1, axis = 1)
        print(classification_report(y_val_pred2, val_labels, digits=4))


    def visualize_predictions(self, test_images, test_labels, pred_labels):
        upper = 10

        for ind in range(upper):
            plt.title("Predicted={}, True={}".format(pred_labels[ind], test_labels[ind]))
            img = test_images[ind].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.show()
