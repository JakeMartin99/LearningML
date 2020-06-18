# Sourced from the blog at https://victorzhou.com/blog/intro-to-neural-networks/
import numpy as np

def mse_loss(y_true, y_pred):
    # Both inputs are numpy arrays of same len
    # Sum of squared diffs, divided by len
    return ((y_true - y_pred) ** 2).mean()

def sigmoid(x):
    # Activation function: f(x) = 1 / (1+e^-x)
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # f'(x) = f(x) * (1 - f(x)) for f(x) = 1 / (1+e^-x)
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use Activation Function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:
    '''
    A neural net with:
        - 2 inputs
        - 1 hidden layer w/ 2 neurons (h1, h2)
        - 1 output layer neuron (o1)
    Each neuron has same weights and bias:
        - w = [0,1]
        - b = 0
    '''
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        #The inputs for o1 are outputs from h1 & h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

class OurNeuralNetwork2:
    '''
    A neural net with:
        - 2 inputs
        - 1 hidden layer w/ 2 neurons (h1, h2)
        - 1 output layer neuron (o1)
    '''
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    #Inefficient but informative
    def train(self, data, all_y_trues):
        '''
        - data is an (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array of n elements
          Elements in all_y_trues correspond to those in data
        '''
        learn_rate = 0.1
        epochs = 1000           #Number of times looping through dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward for later usage
                

weights = np.array([0,1])       # w1 = 0, w2 = 1
bias = 4                        # b = 4
n = Neuron(weights, bias)

x = np.array([2,3])             # x1 = 2, x2 = 3
print(n.feedforward(x))         # Output ~0.999

network = OurNeuralNetwork()
x = np.array([2,3])
print(network.feedforward(x))   # Output ~0.7216

y_true = np.array([1,0,0,1])
y_pred = np.array([0,0,0,0])
print(mse_loss(y_true, y_pred)) # Output 0.5
