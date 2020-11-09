import numpy as np
import struct
import sys

# assuming MNIST dataset file format (big endian)
def dataset_reader(path):
    f = open(path, 'rb')
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))      # reads 4 integers (16 bytes) that are in big-endian format
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>')) # we know that each pixel takes values in [0, 255]
    data = data.reshape((size, rows, cols))
    f.close()

    return (data, size, rows, cols)

# assuming MNIST labels file format (big endian)
def labels_reader(path):
    f = open(path, 'rb')
    magic, size = struct.unpack(">II", f.read(8))
    labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    f.close()

    return labels

def die(error_message, error_code):
    print(error_message, file=sys.stderr)
    sys.exit(error_code)


def ask_for_hyperparameters():
    epochs = int(input("> Enter training epochs: "))
    batch_size = int(input("> Enter training batch size: "))
    kernel_size = int(input("> Enter kernel size: "))
    n_filters = int(input("> Enter the number of filters: "))
    convs_per_layer = int(input("> Enter the number of convolutional nets per layer: "))
    
    return (epochs, batch_size, convs_per_layer, kernel_size, n_filters)