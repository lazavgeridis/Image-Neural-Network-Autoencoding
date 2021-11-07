# Introduction  
This is the 2nd project of the course "Software Development for Algorithmic
Problems". The first part of the project involved implementing an **Autoencoder** Neural Network 
and performing a variety of training experiments, in order to achieve the best
possible values of validation loss and validation accuracy. In the second part
of the project, we reuse the encoder layers of the autoencoder, and add 2 extra layers:
a fully-connected layer and an output layer. The newly constructed model is
a **Convolutional Neural Network**, which is then trained in 2 phases. After
training, we evaluate our model on a validation set and interestingly, it performs very well 
in our classification task.

The dataset used for both parts of the project was once again [MNIST](http://yann.lecun.com/exdb/mnist/), but this time the labels were also used, 
in order to measure the accuracy scores of our models.  

# Part 1 : Autoencoder
The user can choose between either loading a pre-trained Autoencoder model and
re-training or constructing a new Autoencoder model and training from scratch.
Training hyperparameters such as number of epochs, minibatch size, number of
convolutional layers and the kernel size of each conv layer can be adjusted 
interactively by the user. After training, the user can visualize the
training/validation loss of each run. The Autoencoder is trained on the MNIST
training set (60.000 samples), which is split into separate training and validation sets.  

**Note** : The encoder architecture is the exact mirror image of the decoder


# Part 2 : CNN Classifier
Utilizing the Encoder architecture of the Autoencoder model trained and
saved in the previous part, we extend the model by adding an extra
fully-connected and an output layer. The new model is now a CNN classifier and
we train it in 2 phases: during the 1st phase we "freeze" all the layers' weights
except the weights of the fully-connected layer we just added, thus reducing
substantially the required training time. In phase 2, we train the entire
network in an attempt to achieve the best possible performance in terms of 
classification accuracy and loss. After training is completed, the classifier's 
ability to generalize is evaluated using the MNIST test set (10.000 samples). 
The user always has the option of repeating training with different
hyperparameter values (nodes in the fc layer, epochs, minibatch size).


# Execution
For part 1, the program is executed as:  
```
$ python autoencoder.py -d ../datasets/train-images-idx3-ubyte
```

For part 2, the program is executed as:
```
$ python classification.py -d ../datasets/train-images-idx3-ubyte 
                           -dl ../datasets/train-labels-idx1-ubyte
                           -t ../datasets/t10k-images-idx3-ubyte
                           -tl ../datasets/t10k-labels-idx1-ubyte
                           -model ../saved_models/*.h5
```

