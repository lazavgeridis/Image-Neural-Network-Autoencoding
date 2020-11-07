import tensorflow as tf

from keras.layers import Conv2D, MaxiPooling2D


class Encoder():
    
    def __init__(self, batch_size, epochs, n_filters, filter_size, n_layers):
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_layers = n_layers


    def define_model(self):
        pass


    def train_model(self):
        pass
        
