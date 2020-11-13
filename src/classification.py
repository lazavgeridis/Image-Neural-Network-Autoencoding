from sklearn.model_selection import train_test_split
from utils import *
from cnn_classifier import *


def main():
    args = classification_parseargs()
    
    # read files and store the data
    trainset, _, x_dim, y_dim = dataset_reader(args.d)
    trainlabels = labels_reader(args.dl)
    testset, testset_size, _, _ = dataset_reader(args.t)
    testlabels = labels_reader(args.tl)
    
    # rescale pixels in [0, 1]
    trainset = trainset / 255.0
    testset  = testset / 255.0

    # reshape the data, tensors of shape (size, rows, cols, inchannel)
    trainset = trainset.reshape(-1, x_dim, y_dim, 1)
    testset  = testset.reshape(-1, x_dim, y_dim, 1)

    # reserve some training samples for validation
    x_train, x_val, y_train, y_val = train_test_split(trainset, trainlabels, test_size=0.1)

    # CNN Classifier Construction
    ae = load_model(args.model)
    classifier = Classifier(ae)


    # Train CNN Classifier 
    while True:
        fc_nodes = int(input("> Enter number of nodes in fully-connected layer: "))
        ep       = int(input("> Enter training epochs: "))
        batch    = int(input("> Enter training batch size: "))

        classifier.add_layers(fc_nodes)

        classifier.train(x_train, y_train, x_val, y_val, batch, ep)

        print("""\nTraining was completed. You now have the following options:
                1. Repeat the training process with different number of: nodes in fully-connected layer, epochs and batch size
                2. Plot accuracy and loss graphs with respect to the hyperparameters and print evaluation scores (precision, recall, f1)
                3. Classify test set images (using other hyperparams????)
                """)
        
        option = int(input("> Enter the corresponding number: "))
        if option < 1 or option > 3:
            die("\nInvalid option!\n", -2)
        if option != 1:
            break
        else:
            # remove the last 4 layers of the model, since user might request 
            # different number of nodes in the fully-connected layer
            classifier.remove_layers()

    # plot accuracy, loss
    if option == 2:
        classifier.plot_train_curve(x_val, y_val, batch, fc_nodes)

    # classification
    else:
        classifier.test(testset, testlabels, testset_size)
        

if __name__ == '__main__':
    main()
