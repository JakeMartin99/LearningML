# Sourced from the blog at https://victorzhou.com/blog/intro-to-rnns/
import numpy as np
from numpy.random import randn
from data import train_data, test_data
import random

def createInputs(text):
    '''
    Returns array of one-hot vectors representing the words
    in the input text string.
    - text is a string
    - Each one-hot vector has shape (vocab_size, 1)
    '''
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

class RNN:
    # A Vanilla Recurrent NN
    def __init__(self, input_size, output_size, hidden_size=64):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        '''
        Perform a forward pass of the the RNN using given inputs
        Returns the final output and hidden state
        - inputs is an array of one-hot vectors with shape (input_size, 1)
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = { 0: h }

        # Perform each RNN step
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i+1] = h

        y = self.Why @ h + self.by
        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        '''
        Perform a backward pass of the RNN
        - d_y (dL/dy) has shape (output_size, 1)
        - learn_rate is a float
        '''
        n = len(self.last_inputs)

        # Calc dL/dWhy and dL/dby
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Initialize dL/dWhh, dL/dWxh, dL/dbh to 0
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Calc dL/dh for the last h
        d_h = self.Why.T @ d_y

        # Backprop through time
        for t in reversed(range(n)):
            # An intermed val: dL/dh * (1 - h^2)
            temp = (1 - self.last_hs[t+1] ** 2) * d_h

            # dL/db = dL/dh * (1-h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1-h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1-h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T

            # Next dL/dh = dL/dh * (1-h^2) * Whh
            d_h = self.Whh @ temp

        # Clip to prevent exploding gradients
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using grad descent
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

def softmax(xs):
    # Applies Softmax to the input array
    return np.exp(xs) / sum(np.exp(xs))

def processData(data, backprop=True):
    '''
    Return RNN loss and accuracy for given data
    - data is a dictionary mapping text to True / False
    - backprop determines if backward phase should run
    '''
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        # Forward
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Calc loss / accuracy
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Build dL/dy
            dL_dy = probs
            dL_dy[target] -= 1
            # Backward
            rnn.backprop(dL_dy)

    return loss / len(data), num_correct / len(data)

# Create the vocabulary
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

# Assign indices to each words
word_to_idx = { w: i for i, w in enumerate(vocab)}
idx_to_word = { i: w for i, w in enumerate(vocab)}
# print(word_to_idx['good']) # 16 (may change)
# print(idx_to_word[0]) # 'sad' (may change)

# Init the RNN
rnn = RNN(vocab_size, 2)

# Training loop
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch+1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

# Allow user sentences -- Addition wholly by Jake Martin
running = True
while running:
    sentence = createInputs(input("\nTry a sentence: "))
    output, _ = rnn.forward(sentence)
    probs = softmax(output)
    print("Prob Good: ", round(probs[1][0]*100,2), "%" )
    print("Prob Bad: ", round(probs[0][0]*100,2), "%" )

    cont = input("Run again? (y/n): ")
    if cont == 'n':
        running = False
