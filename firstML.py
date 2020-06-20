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
        epochs = 1000           # Number of times looping through dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward for later usage
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calc partial deris
                dL_dypred = -2 * (y_true - y_pred)

                # Neuron o1
                dypred_dw5 = h1 * deriv_sigmoid(sum_o1)
                dypred_dw6 = h2 * deriv_sigmoid(sum_o1)
                dypred_db3 = deriv_sigmoid(sum_o1)
                dypred_dh1 = self.w5 * deriv_sigmoid(sum_o1)
                dypred_dh2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                dh1_dw1 = x[0] * deriv_sigmoid(sum_h1)
                dh1_dw2 = x[1] * deriv_sigmoid(sum_h1)
                dh1_db1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                dh2_dw3 = x[0] * deriv_sigmoid(sum_h2)
                dh2_dw4 = x[1] * deriv_sigmoid(sum_h2)
                dh2_db2 = deriv_sigmoid(sum_h2)

                # --- Update weights and Biases
                # Neuron h1
                self.w1 -= learn_rate * dL_dypred * dypred_dh1 * dh1_dw1
                self.w2 -= learn_rate * dL_dypred * dypred_dh1 * dh1_dw2
                self.b1 -= learn_rate * dL_dypred * dypred_dh1 * dh1_db1

                # Neuron h2
                self.w3 -= learn_rate * dL_dypred * dypred_dh2 * dh2_dw3
                self.w4 -= learn_rate * dL_dypred * dypred_dh2 * dh2_dw4
                self.b2 -= learn_rate * dL_dypred * dypred_dh2 * dh2_db2

                # Neuron o1
                self.w5 -= learn_rate * dL_dypred* dypred_dw5
                self.w6 -= learn_rate * dL_dypred * dypred_dw6
                self.b3 -= learn_rate * dL_dypred * dypred_db3

            # --- Calculate total loss for the epochs
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

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

#Define dataset
data = np.array([
    [-2, -1],   #Alice
    [25, 6],    #Bob
    [17, 4],    #Charlie
    [-15, -6],  #Diana
])
all_y_trues = np.array([
    1,      #Alice
    0,      #Bob
    0,      #Charlie
    1,      #Diana
])

#Train OurNeuralNetwork2
print("\n\n\n")
network = OurNeuralNetwork2()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
me = np.array([50, 12])    # 185 lbs, 6' = 72"
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
print("Me: %.3f" % network.feedforward(me))       #
