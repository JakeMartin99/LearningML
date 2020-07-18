import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

print(train_images.shape) # (60000, 784)
print(test_images.shape) # (10000, 784)

model = Sequential([
    # Layers...
])
