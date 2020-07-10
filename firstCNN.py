import numpy as np
import mnist

class Conv3x3:
    # A convolution layer using 3x3 filters

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimenstions (num_filters,
        # 3, 3)  Divide by 9 to reduce variance of vals
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions w/valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Perform forward pass of the conv layer using given input.
        Return 3d numpy array of dimen. (h, w, num_filters).
        - input is a 2d numpy array
        '''
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region * self.filters, axis = (1,2))

        return output

    def backprop(self, dL_dout, learn_rate):
        '''
        Perform backprop of the conv layer
        - dL_dout is the loss gradient for this layer's outputs
        - learn_rate is a float
        '''
        dL_dfilters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dL_dfilters[f] += dL_dout[i,j,f] * im_region

        # Update filters
        self.filters -= learn_rate * dL_dfilters

        # Nothing to return since this is the input layer
        return None

class MaxPool2:
    # A max pooling layer using a pool size of 2 (2x2 square)

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions for pooling
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs forward pass of maxpool2 layer w/given input.
        Returns 3d numpy array w/dimen. (h/2, w/2, num_filters).
        - input is a 3d numpy array w/dimen. (h, w, num_filters)
        '''
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region, axis=(0,1))

        return output

    def backprop(self, dL_dout):
        '''
        Perform backprop of maxpool layer
        Returns loss gradient for this layer's inputs
        - dL_dout is the loss gradient for this layer's outputs
        '''
        dL_dinput = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If pixel was the max, copy gradient to it
                        if im_region[i2, j2, f2] == amax[f2]:
                            dL_dinput[i*2 +i2, j*2 + j2, f2] = dL_dout[i,j,f2]

        return dL_dinput

class Softmax:
    # A std, fully-connected layer w/softmax activation

    def __init__(self, input_len, nodes):
        # Divide by input_len to reduce init val variance
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs forward pass of softmax layer w/given input.
        Returns a 1d numpy array of respective prob values.
        - input can be any array of any dimensions
        '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, dL_dout, learn_rate):
        '''
        Performs a backward pass of the softmax layer
        Returns the loss gradient for this layers inputs
        - dL_dout is the loss gradient for this layers outputs
        - learn_rate is a float
        '''
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)

            dout_dt = -t_exp[i] * t_exp / (S ** 2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = self.weights
            dL_dt = gradient * dout_dt
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt

            self.weights -= learn_rate * dL_dw
            self.biases -= learn_rate * dL_db
            return dL_dinputs.reshape(self.last_input_shape)

# The mnist package handles the MNIST dataset
# Images are of handwritten digits
# Using only first 1k / 10k for all for time
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.train_images()[:1000]
test_labels = mnist.train_labels()[:1000]

conv = Conv3x3(8)
output = conv.forward(train_images[0])
print(output.shape) # (26, 26, 8)

pool = MaxPool2()
output = pool.forward(output)
print(output.shape) # (13, 13, 8)

softmax = Softmax(13*13*8, 10) # 13x13x8 -> 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calcs
    accuracy and cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # Image is trnsfrm from [0, 255] to [-0.5, 0.5] for ease
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    '''
    Completes a full training step on given image + label
    Returns cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)
    # Initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    #Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch+1))
    # Shuffle training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Training
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        # Print stats every 100 steps
        if i>0 and i%100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i+1, loss/100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
