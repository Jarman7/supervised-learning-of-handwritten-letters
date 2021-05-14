# Tom Jarman, 13/03/21

import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat


###########################
#                         #
#  Activation Functions   #
#                         #
###########################
def relu(h):
    """ ReLU function, takes input h and return result of ReLU(h) """
    return np.maximum(h, 0)



def relu_prime(h):
    """ Derivative of the ReLU function """
    return_value = h
    return_value[return_value <= 0] = 0
    return_value[return_value > 0 ] = 1
    return return_value



###########################
#                         #
#       Loading Data      #
#                         #
###########################
def load_data(filename):
    """ Loads dataset and performs neccessary spliting, shuffling and formatting """
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
    train_images = combined_training[:20800,:-1] / 255 # Normalize data, values are now between 0 and 1
    train_labels = combined_training[:20800,-1][...,None] # Turns back into column vector
    validation_images = combined_training[20800:,:-1] / 255 # Normalize data, values are now between 0 and 1
    validation_labels = combined_training[20800:,-1][...,None] # Turns back into column vector

    # Load training images and labels
    test_images = emnist['test_images'] / 255 # Normalize data, values are now between 0 and 1
    test_labels = emnist['test_labels']

    return train_images, train_labels, test_images, test_labels, validation_images, validation_labels



###########################
#                         #
#      Init Functions     #
#                         #
###########################
def init_expected_outputs(data, no_labels=26):
    """ Takes in output labels and converts to corresponding output layer output"""
    expected_outputs = np.zeros((data.shape[0], no_labels))
    
    for i in range(0,data.shape[0]):    
        expected_outputs[i, data[i].astype(int)]=1

    return expected_outputs



def init_weights(n_input_layer, n_hidden_layer, n_hidden_layer_2, n_output_layer, xavier_init):
    """ Initialises weights depending on layer sizes and whether Xavier Initialisation is needed
        Reference: Takes inspiration from COM3240 Lab 2 - Matthew Ellis, 15/02/2021
    """
    W1, W2, W3 = None, None, None
    
    if xavier_init: # Checks if Xavier initialisation is wanted
        # Initialises weights depending on number of layers present using:
        # Normally distributed random number * square_root(1 / number of input neurons to that layer)
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
        # Weights are randomly picked from a uniform distribution between 0 and 1
        # They are normalized by making sure the weights sum to 1
        # Uses different configurations depending on number of layers required
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
    """ Initialises the bias weights (thresholds) for each neuron in each layer"""
    bias_W1, bias_W2, bias_W3 = None, None, None

    # Create empty arrays of the desired size given by the number of neurons per layer
    # Arrays are populated with 0's
    if n_hidden_layer > 0:
        bias_W1 = np.zeros((n_hidden_layer,1))

        if n_hidden_layer_2 > 0:
            bias_W2=np.zeros((n_hidden_layer_2,1))    
            bias_W3=np.zeros((n_output_layer,1))

        else:
            bias_W2 = np.zeros((n_output_layer,1))

    else:
        bias_W1 = np.zeros((n_output_layer,1))

    return bias_W1, bias_W2, bias_W3



###########################
#                         #
#     Training Metrics    #
#                         #
###########################
def calculate_average_weight(tau, average_weight, average_weight_plot, prev_w1, w1, epoch):
    """ Calculates average change """
    if epoch == 0:
        average_weight = w1 - prev_w1 # When Epoch = 0 using delta_w

    else:
        delta_w1 = w1 - prev_w1
        average_weight = (average_weight * (1 - tau)) + (tau * delta_w1)
    
    average_weight_plot[epoch] = np.sum(average_weight)
    print("Average weight: {}".format(np.sum(average_weight)))
    return average_weight_plot, average_weight

    

def plot_results(data, xlabel, ylabel, title, legend):
    plt.plot(data, label=legend)
    plt.legend(loc="upper left")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



###########################
#                         #
#      Training Model     #
#                         #
###########################
def train(epoch, w1, w2, w3, samples, n_batches, bias_w1, bias_w2, bias_w3, n_hidden_layer, n_hidden_layer_2, 
          batch_size, train_data, train_output, valid_data, valid_output, learning_rate, lmbda, l1):
    """ 
    Trains the model based on the system parameters
    Uses the ReLU function and derivative to train neuron weights
    Weights are trained depending on the number of layers
    Batch training is carried out with weights being updated at the end of each bach
    After each epoch accuracy, error and average weight update (for single layer) are calculated
    At the end of training graphs are output for error, accuracy and average weight (single layer) per epoch
    """
    # Initialise empty error and accuracy arrays
    errors = np.zeros((epoch,))
    accuracies = np.zeros((epoch,))

    # If it is only a single layer network initialise variables for calcualting average weight
    if (n_hidden_layer == 0) and (n_hidden_layer_2 == 0):
        tau = 0.01
        average_weight = np.zeros(w1.shape)
        average_weight_plot = np.zeros((epoch,1))
        prev_w1 = np.copy(w1)

    # Epoch loop
    for i in range(epoch):
        # Build an array of shuffled indexes
        shuffled_indexes = np.random.permutation(samples)

        # Batch loop
        for batch in range(0, n_batches):
            
            # Initialise empty change in weight and bias depending on number of layers
            delta_w1 = np.zeros(w1.shape)
            delta_bias_w1 = np.zeros(bias_w1.shape)
            if n_hidden_layer > 0:
                delta_w2 = np.zeros(w2.shape)
                delta_bias_w2 = np.zeros(bias_w2.shape)
                if n_hidden_layer_2 > 0:
                    delta_w3 = np.zeros(w3.shape)
                    delta_bias_w3 = np.zeros(bias_w3.shape)

            # Extract indexes, and corresponding data from the input and expected output
            indexes = shuffled_indexes[batch*batch_size : (batch+1)*batch_size]
            x0 = train_data[indexes].T
            t = train_output[indexes].T

            # Apply input weights to summation of inputs and add bias terms
            h1 = np.matmul(w1, x0) + bias_w1
            # Apply the activation function to the summation
            x1 = relu(h1)
            
            # For first hidden layer
            if n_hidden_layer > 0:
                # Apply input weights to summation of inputs and add bias terms
                h2 = np.matmul(w2, x1) + bias_w2
                # Apply the activation function to the summation
                x2 = relu(h2)

                # For second hidden layer
                if n_hidden_layer_2 > 0:
                    # Apply input weights to summation of inputs and add bias terms
                    h3 = np.matmul(w3, x2) + bias_w3
                    # Apply the activation function to the summation
                    x3 = relu(h3)

                    # Error signal
                    error = t - x3
                    # Local gradient for second hidden layer
                    delta_3 = relu_prime(x3) * error
                    # Change in weight at second hidden layer
                    delta_w3 = (learning_rate / batch_size) * np.matmul(delta_3, x2.T)
                    # Change in bias at second hidden layer
                    delta_bias_w3 = (learning_rate / batch_size) * np.sum(delta_3, axis=1)
                    # Reshape to be a matrix rather than column vector
                    delta_bias_w3 = delta_bias_w3.reshape(-1, 1)

                    # Local gradient for first hidden layer
                    delta_2 = relu_prime(h2) * np.matmul(w3.T, delta_3)
                    # Change in weight at first hidden layer
                    delta_w2 = (learning_rate / batch_size) * np.matmul(delta_2, x1.T)
                    # Change in bias at first hidden layer
                    delta_bias_w2 = (learning_rate / batch_size) * np.sum(delta_2, axis=1)
                    # Reshape to be a matrix rather than column vector
                    delta_bias_w2 = delta_bias_w2.reshape(-1, 1)


                    # Local gradient for input layer
                    delta_1 = relu_prime(h1) * np.matmul(w2.T, delta_2)
                    # Change in weight at input layer
                    delta_w1 = (learning_rate / batch_size) * np.matmul(delta_1, x0.T)
                    # Change in bias at input layer
                    delta_bias_w1 = (learning_rate / batch_size) * np.sum(delta_1, axis=1)
                    # Reshape to be a matrix rather than column vector    
                    delta_bias_w1 = delta_bias_w1.reshape(-1, 1)


                else:
                    # Error signal
                    error = t - x2
                    # Change in weight at first hidden layer
                    delta_2 = relu_prime(x2) * error
                    # Change in weight at first hidden layer
                    delta_w2 = (learning_rate / batch_size) * np.matmul(delta_2, x1.T)
                    # Change in bias at first hidden layer
                    delta_bias_w2 = (learning_rate / batch_size) * np.sum(delta_2, axis=1)
                    # Reshape to be a matrix rather than column vector
                    delta_bias_w2 = delta_bias_w2.reshape(-1, 1)

                    # Local gradient for input layer
                    delta_1 = relu_prime(h1) * np.matmul(w2.T, delta_2)
                    # Change in weight at input layer
                    delta_w1 = (learning_rate / batch_size) * np.matmul(delta_1, x0.T)
                    # Change in bias at input layer
                    delta_bias_w1 = (learning_rate / batch_size) * np.sum(delta_1, axis=1)
                    # Reshape to be a matrix rather than column vector    
                    delta_bias_w1 = delta_bias_w1.reshape(-1, 1)

            else:
                # Error signal
                error = t - x1
                # Local gradient for input layer
                delta_1 = relu_prime(x1) * error
                # Change in weight at input layer
                delta_w1 = (learning_rate / batch_size) * np.matmul(delta_1, x0.T)
                # Change in bias at input layer
                delta_bias_w1 = (learning_rate / batch_size) * np.sum(delta_1, axis=1)
                # Reshape to be a matrix rather than column vector    
                delta_bias_w1 = delta_bias_w1.reshape(-1, 1)

            # Checks if L1 error is used as well
            if l1:
                # Takes away the derivative of L1 from the change in weight
                delta_w1 -= (learning_rate / batch_size) * lmbda * np.sign(w1)
                # Takes away the derivative of L1 from the change in bias
                delta_bias_w1 -= (learning_rate / batch_size) * lmbda * np.sign(bias_w1)

                # Checks if hidden layer present
                if n_hidden_layer > 0:
                    # Takes away the derivative of L1 from the change in weight
                    delta_w2 -= (learning_rate / batch_size) * lmbda * np.sign(w2)
                    # Takes away the derivative of L1 from the change in bias
                    delta_bias_w2 -= (learning_rate / batch_size) * lmbda * np.sign(bias_w2)
                    
                    # Checks if second hidden layer present
                    if n_hidden_layer_2 > 0:
                        # Takes away the derivative of L1 from the change in weight
                        delta_w3 -= (learning_rate / batch_size) * lmbda * np.sign(w3)
                        # Takes away the derivative of L1 from the change in bias
                        delta_bias_w3 -= (learning_rate / batch_size) * lmbda * np.sign(bias_w3)


            # Add change in weight
            w1 += delta_w1
            # Add change in bias
            bias_w1 += delta_bias_w1

            # Checks if hidden layer present
            if n_hidden_layer > 0:
                # Add change in weight
                w2 += delta_w2
                # Add change in bias
                bias_w2 += delta_bias_w2
        
                # Checks if second hidden layer present
                if n_hidden_layer_2 > 0:
                    # Add change in weight
                    w3 += delta_w3
                    # Add change in bias
                    bias_w3 += delta_bias_w3

        # Calculate and print average weight (single layer), accuracy and error at the end of the epoch
        print("------ Epoch {} ------".format(i+1))
        if n_hidden_layer == 0:
            # If single layer present calculate average weight change
            average_weight_plot, average_weight = calculate_average_weight(tau, average_weight, average_weight_plot,
                                                                           prev_w1, w1, i)
            prev_w1 = np.copy(w1)
        # Calculate accuracy and error based on validation data
        accuracies[i], errors[i] = test(valid_data, valid_output, n_hidden_layer, n_hidden_layer_2, w1, w2, w3, 
                                        bias_w1, bias_w2, bias_w3, l1, lmbda)
        print("---------------------")
        print("\n")
    
    # Plot results for error, accruacy and average weight (single layer)
    #if n_hidden_layer == 0:
    #    plot_results(average_weight_plot, 'Epoch', 'Average Weight Update Sum',
    #                 'Average Weight Update Sum per Epoch', 'Average Weight Update Sum')
    #plot_results(errors, 'Epoch', 'Error', 'Error on Validation Set per Epoch', 'Error')
    #plot_results(accuracies, 'Epoch', 'Accuracy', 'Accuracy on Validation Set per Epoch', 'Accuracy')
    return w1, w2, w3, bias_w1, bias_w2, bias_w3



def test(test_data, test_output, n_hidden_layer, n_hidden_layer_2, w1, w2, w3, bias_w1, bias_w2, bias_w3, l1, lmbda):
    """ 
    Predicts outputs depending on input data, trained weights and biases
    """
    # Set up initial variables
    samples = test_data.shape[0]
    correct_values = np.argmax(test_output, axis=1)
    predicted_values = np.zeros((samples,))
    error = np.zeros(test_output.shape)
    error_l1 = 0

    # Extract inputs
    x0 = test_data.T

    # Apply input weights to summation of inputs and add bias terms
    h1 = np.matmul(w1, x0) + bias_w1
    # Apply the activation function to the summation
    x1 = relu(h1)

    # Checks if L1 is wanted
    if l1:
        # Calculates l1 error for input layer
        error_l1 = lmbda * np.sum(np.sqrt(np.square(w1)))

    # Checks if hidden layer is needed
    if n_hidden_layer > 0:
        # Apply input weights to summation of inputs and add bias terms
        h2 = np.matmul(w2, x1) + bias_w2
        # Apply the activation function to the summation
        x2 = relu(h2)
        if l1:
            # Calculates l1 error for hidden layer
            error_l1 += lmbda * np.sum(np.sqrt(np.square(w2)))

        if n_hidden_layer_2 > 0:
            # Apply input weights to summation of inputs and add bias terms
            h3 = np.matmul(w3, x2) + bias_w3
            # Apply the activation function to the summation
            x3 = relu(h3)
            if l1:
                # Calculates l1 error for second hidden layer
                error_l1 += lmbda * np.sum(np.sqrt(np.square(w3)))

            # Calculate labels
            predicted_values = np.argmax(x3, axis=0)
            # Error Signal
            error = (test_output - x3.T)

        else:
            # Calculate labels
            predicted_values = np.argmax(x2, axis=0)
            # Error Signal
            error = (test_output - x2.T)

    else:
        # Calculate labels
        predicted_values = np.argmax(x1, axis=0)
        # Error Signal
        error = (test_output - x1.T)

    # Calculate MSE error
    error_mse = np.sum(np.square(error)) / (2 * error.shape[0])

    # Add MSE error to L1 error, if L1 isn't used this will add 0
    error = error_mse + error_l1

    # Calculate accuracy of predictions
    accuracy = (np.sum(predicted_values == correct_values) / samples) * 100

    print("Accuracy = ", accuracy)
    print("Error = ", error)
    return accuracy, error



def main():
    # Load the EMNIST dataset
    train_images, train_labels, test_images, test_labels, validation_images, validation_labels = load_data('emnist-letters-1k.mat')

    # Systems Variables
    NO_LABELS = 26
    EPOCH = 250
    LAMBDA = 0.00001 
    BATCH_SIZE = 50
    SAMPLES = train_images.shape[0]
    IMAGE_SIZE = train_images.shape[1]
    N_BATCHES = int(math.ceil(SAMPLES / BATCH_SIZE))
    LEARNING_RATE = 0.05
    XAVIER_INIT = True # Boolean, determines if XAVIER_INIT is used
    L1_ERROR = True # Boolean, determines if L1 Error is used

    # Number of neurons in each layer
    # For the hidden layers 0 means the layer doesn't exist
    N_INPUT_LAYER = IMAGE_SIZE
    N_HIDDEN_LAYER = 100
    N_HIDDEN_LAYER_2 = 20
    N_OUTPUT_LAYER = NO_LABELS

    # Turns labels into expected output of output layer
    test_output = init_expected_outputs(test_labels)
    train_output = init_expected_outputs(train_labels)
    validation_output = init_expected_outputs(validation_labels)

    # Initialises weights before training
    w1, w2, w3 = init_weights(N_INPUT_LAYER, N_HIDDEN_LAYER, N_HIDDEN_LAYER_2, N_OUTPUT_LAYER, XAVIER_INIT)
    # Initialises Bias terms
    bias_w1, bias_w2, bias_w3 = init_bias(N_HIDDEN_LAYER, N_HIDDEN_LAYER_2, N_OUTPUT_LAYER)

    # Trains model and returns weights and bias terms
    w1, w2, w3, bias_w1, bias_w2, bias_w3 = train(EPOCH, w1, w2, w3, SAMPLES, N_BATCHES, bias_w1, bias_w2, bias_w3,
                                                  N_HIDDEN_LAYER, N_HIDDEN_LAYER_2, BATCH_SIZE, train_images, 
                                                  train_output, validation_images, validation_output, LEARNING_RATE,
                                                  LAMBDA, L1_ERROR)

    # Tests data based on system parameters and trained model, prints accuracy and error
    print("------ Test Data ------")
    test(test_images, test_output, N_HIDDEN_LAYER, N_HIDDEN_LAYER_2, w1, w2, w3, bias_w1, 
         bias_w2, bias_w3, L1_ERROR, LAMBDA)
    print("-----------------------")
    print("\n")

if __name__ == "__main__":
    main()