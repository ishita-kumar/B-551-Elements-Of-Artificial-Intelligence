import pickle
from random import seed

import numpy as np
import pandas as pd


# Initialize a network
def initialize_network(n_inputs, n_hidden1, n_hidden2, n_outputs):
    """
    :param n_inputs: number nodes in input layer
    :param n_hidden1: number nodes in hidden layer
    :param n_hidden2:number nodes in hidden layer
    :param n_outputs:number nodes in output layer
    :return:
    """
    seed(42)
    network = []
    hidden_layer1 = np.transpose(
        np.random.randn(n_inputs + 1, n_hidden1) * np.sqrt(2 / n_hidden1))
    network.append(hidden_layer1)
    hidden_layer2 = np.transpose(
        np.random.randn(n_hidden1 + 1, n_hidden2) * np.sqrt(2 / n_hidden2))
    network.append(hidden_layer2)
    output_layer = np.transpose(
        np.random.randn(n_hidden2 + 1, n_outputs) * np.sqrt(2 / n_outputs))
    network.append(output_layer)

    return network


def softmax(A):
    """
    :param A: matrix
    :return: returns softmax
    """
    expA = np.exp(A)
    return np.divide(expA, expA.sum(axis=1, keepdims=True))


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def feed_forward(W, bias, X):
    """
    :param W:Weights
    :param bias: bias
    :param X:Input Data
    :return:softmax output
    """
    x = X
    i = 0
    outputs_layer = [[], [], []]
    z = [[], [], []]
    for layer in W:
        a = np.dot(x, W[i]) + bias[i]
        z[i] = a
        if i == 0 or i == 1:
            x = sigmoid(a)
            outputs_layer[i] = x
        elif i == 2:
            x = softmax(a)
            outputs_layer[i] = x
        i += 1
    return (outputs_layer, z)


def sigmoid_derv(s):
    """
    :param s:
    :return: derivative of sigmoid function
    """

    return s * (1 - s)


def cross_entropy(pred, real):
    """
    :param pred: predicted value
    :param real: true values
    :return:
    """
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def delta_cross_entropy(X, y):
    """
    :param X: is the output from fully connected layer (num_examples x num_classes)
    :param y: is labels (num_examples x 1)
    """
    y = y.argmax(axis=1)
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


def backpropagation(W, bias, a, z, X, y_one_hot):
    """
    :param W: Weight
    :param bias: Bias
    :param a: activations
    :param z: z = W.T*X +bias
    :param X:Data
    :param y_one_hot: One hot encoded true labels
    :return: returns updated weights and bias
    """

    lr = 1
    a3_delta = cross_entropy(a[2], y_one_hot)  # w3
    z2_delta = np.dot(a3_delta, W[2].T)
    a2_delta = z2_delta * sigmoid_derv(a[1])  # w2
    z1_delta = np.dot(a2_delta, W[1].T)
    a1_delta = z1_delta * sigmoid_derv(a[0])  # w1
    W[2] -= lr * np.dot(a[1].T, a3_delta)
    bias[2] -= lr * np.sum(a3_delta, axis=0)
    W[1] -= lr * np.dot(a[0].T, a2_delta)
    bias[1] -= lr * np.sum(a2_delta, axis=0)
    W[0] -= lr * np.dot(X.T, a1_delta)
    bias[0] -= lr * np.sum(a1_delta, axis=0)
    return W, bias


def train(W, b, X, y_one_hot, model_file):
    """
    :param W: Weights
    :param b:Bias
    :param X:Data

    This function iterativel calls feed forward and backpropagation
    """
    Weight = W
    Bias = b
    data = X
    for epoch in range(1000):
        (activation, z) = feed_forward(Weight, Bias, data)
        (Weight_new, Bias_new) = backpropagation(Weight, Bias, activation, z,
                                                 data, y_one_hot)
        Weight = Weight_new
        Bias = Bias_new
    with open(model_file, 'wb') as fp:  ## write the weights and bias to a file
        pickle.dump((Weight, Bias), fp)

    return (Weight, Bias, activation)


'''
This here is the driver function where in train we initialise the network and the weights,bias.
'''


def nn(mode, mode_file, model_file):
    """
    :param mode: test or train
    :param mode_file: test or train file
    :param model_file: stored weights and bias
    :return:
    """
    if mode == 'train':
        data = np.genfromtxt(mode_file)
        input_data = np.delete(data, 0, 1)
        Y = input_data[..., 0]
        X = np.delete(input_data, 0, 1)
        network = initialize_network(192, 128, 64, 4)
        y_one_hot = np.array(pd.get_dummies(Y))
        bias = []
        W = []
        for layer in network:
            bias.append(layer[..., -1])
            weights = np.vstack(layer)
            W.append(np.delete(weights, -1, 1).T)

        train(W, bias, X, y_one_hot, model_file)
        (Weight, Bias, activation) = train(W, bias, X, y_one_hot, model_file)
        t_hat = activation[-1]

        prediction = t_hat.argmax(axis=-1)
        target = y_one_hot.argmax(axis=-1)

        return round(np.mean(prediction == target) * 100, 2)

    else:
        with open(model_file, 'rb') as fp:
            (Weight, Bias) = pickle.load(fp)
        test_data = np.genfromtxt(mode_file)
        filename = np.loadtxt(mode_file, dtype='str')[..., 0]
        test_data = np.delete(test_data, 0, 1)
        test_labels = test_data[..., 0]
        test_data = np.delete(test_data, 0, 1)
        test_target_one_hot = np.array(pd.get_dummies(test_labels))
        (t, zz) = feed_forward(Weight, Bias, test_data)
        pred = t[-1]
        test_prediction = pred.argmax(axis=-1)
        test_target = test_target_one_hot.argmax(axis=-1)

        orientation = np.where(test_prediction == 1, 90, test_prediction)
        orientation = np.where(orientation == 2, 180, orientation)
        orientation = np.where(orientation == 3, 270, orientation)

        output_file = np.column_stack((filename, orientation))
        f = open("nn/output.txt", "w")
        f.write("\n".join(
            str(elem[0]) + ' ' + str(elem[1]) for elem in output_file))
        f.close()

        return round(np.mean(test_prediction == test_target) * 100, 2)

