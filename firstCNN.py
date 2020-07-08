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

# The mnist package handles the MNIST dataset
# Images are of handwritten digits
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(8)
output = conv.forward(train_images[0])
print(output.shape) # (26, 26, 8)
