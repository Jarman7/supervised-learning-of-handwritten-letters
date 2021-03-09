# Tom Jarman

import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat

def relu(h):
    """ ReLU function, takes input h and return result of ReLU(h) """
    return max(h, 0)


def load_data(filename):
    """ Loads dataset and performs neccessary spliting, shuffling and formatting """
     # Load in EMNIST dataset
    emnist = loadmat(filename)

    # Load training images and labels
    train_images_unshuffled = emnist['train_images']
    train_labels_unshuffled = emnist['train_labels']

    # Combine labels and training data
    combined_training = np.hstack((train_images_unshuffled, train_labels_unshuffled))

    # Shuffle data
    np.random.shuffle(combined_training)

    # Seperate into data and labels
    # Split into training and validation sets
    train_images = combined_training[:20800,:-1]
    train_labels = combined_training[:20800,-1][...,None] # Turns back into column vector
    validation_images = combined_training[20800:,:-1]
    validation_labels = combined_training[20800:,-1][...,None] # Turns back into column vector

    # Load training images and labels
    test_images = emnist['test_images']
    test_labels = emnist['test_labels']

    return train_images, train_labels, test_images, test_labels, validation_images, validation_labels


def init_expected_outputs(data, no_labels=26):
    # Create expected output of output neurons
    expected_outputs = np.zeros((data.shape[0], no_labels))
    
    for i in range(0,data.shape[0]):    
        expected_outputs[i, data[i].astype(int)]=1

    return expected_outputs


def init_weights(n_input_layer, n_hidden_layer, n_hidden_layer_2, n_output_layer, xavier_init):
    W1, W2, W3 = None, None, None
    if xavier_init:
        if n_hidden_layer > 0:
            W1 = np.random.randn(n_hidden_layer, n_input_layer) * np.sqrt(1 / (n_input_layer))

            if n_hidden_layer_2 > 0:
                W2 = np.random.randn(n_hidden_layer_2, n_hidden_layer) * np.sqrt(1 / (n_hidden_layer))
                W3 = np.random.randn(n_output_layer, n_hidden_layer_2) * np.sqrt(1 / (n_hidden_layer_2))

            else:
                W2 = np.random.randn(n_output_layer, n_hidden_layer) * np.sqrt(1 / (n_hidden_layer))

        else:
            W1 = np.random.randn(n_output_layer, n_input_layer) * np.sqrt(1 / (n_input_layer))

    else:
        if n_hidden_layer > 0:
            W1 = np.random.uniform(0,1,(n_hidden_layer, n_input_layer))
            W1 = np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))
            
            if n_hidden_layer_2 > 0:
                W2=np.random.uniform(0,1,(n_hidden_layer_2,n_hidden_layer))
                W2=np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))

                W3=np.random.uniform(0,1,(n_output_layer,n_hidden_layer_2))
                W3=np.divide(W3,np.matlib.repmat(np.sum(W3,1)[:,None],1,n_hidden_layer_2))

            else:
                W2 = np.random.uniform(0,1,(n_output_layer, n_hidden_layer))
                W2 = np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))

        else:
            W1 = np.random.randn(n_output_layer, n_input_layer) * np.sqrt(1 / (n_input_layer))

    return W1, W2, W3


def init_bias(n_hidden_layer, n_hidden_layer_2, n_output_layer):
    bias_W1, bias_W2, bias_W3 = None, None, None

    if n_hidden_layer > 0:
        bias_W1 = np.zeros((n_hidden_layer,))

        if n_hidden_layer_2 > 0:
            bias_W2=np.zeros((n_hidden_layer_2,))    
            bias_W3=np.zeros((n_output_layer,))

        else:
            bias_W2 = np.zeros((n_output_layer,))

    else:
        bias_W1 = np.zeros((n_output_layer))

    return bias_W1, bias_W2, bias_W3

def main():
    train_images, train_labels, test_images, test_labels, validation_images, validation_labels = load_data('emnist-letters-1k.mat')

    # Systems Variables
    NO_LABELS = 26
    EPOCH = 100
    BATCH_SIZE = 50
    SAMPLES = train_images.shape[0]
    IMAGE_SIZE = train_images.shape[1]
    N_BATCHES = int(math.ceil(SAMPLES / BATCH_SIZE))
    LEARNING_RATE = 0.5
    XAVIER_INIT = True

    # Number of neurons in each layer, for the hidden layers 0 means the layer doesn't exist
    N_INPUT_LAYER = IMAGE_SIZE
    N_HIDDEN_LAYER = 200
    N_HIDDEN_LAYER_2 = 0
    N_OUTPUT_LAYER = NO_LABELS

    test_output = init_expected_outputs(test_labels)
    train_output = init_expected_outputs(train_labels)
    validation_output = init_expected_outputs(validation_labels)

    w1, w2, w3 = init_weights(N_INPUT_LAYER, N_HIDDEN_LAYER, N_HIDDEN_LAYER_2, N_OUTPUT_LAYER, XAVIER_INIT)
    bias_w1, bias_w2, bias_w3 = init_bias(N_HIDDEN_LAYER, N_HIDDEN_LAYER_2, N_OUTPUT_LAYER)

if __name__ == "__main__":
    main()