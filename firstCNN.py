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
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region * self.filters, axis = (1,2))

        return output

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
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region, axis=(0,1))

        return output

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
        input = input.flatten()
        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

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

print('MNIST CNN initialized!')
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # Print stats every 100 steps
    if i%100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i+1, loss/100, num_correct)
        )
        loss = 0
        num_correct = 0
