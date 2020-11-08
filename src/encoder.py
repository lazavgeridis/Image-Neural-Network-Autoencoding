import tensorflow as tf

from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential


class Encoder():
    
    def __init__(self, batch_size, epochs, n_filters, filter_size, n_layers):
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_layers = n_layers

        self.model = Sequential()


    def encode_model(self, dims):
        
        initial_filters = self.n_filters
        for i in range(0, self.n_layers):
            self.model.add(Conv2D(initial_filters, kernel_size=self.filter_size, activation="relu", padding='same', input_shape=(dims[0], dims[1], 1), name="conv"+str(i)))
            self.model.add(BatchNormalization(name="batch"+str(i)))
            self.model.add(MaxPool2D(name="pool"+str(i)))

            initial_filters *= 2
        
        return self.model

        
