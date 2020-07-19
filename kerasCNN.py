import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Don't print verbose info msgs
import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Build the model
num_filters = 8
filter_size = 3
pool_size = 2
model = Sequential([
    # Layers...
    Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
    Conv2D(num_filters, filter_size),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
print("\nTraining model:")
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=12,
    validation_data=(test_images, to_categorical(test_labels)),
)

# Save weights
model.save_weights('cnn.h5')
# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images
print("\nPrediction:")
predictions = model.predict(test_images[:5])

# Print the model's predictions
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check predictions against truth
print("Truth:")
print(test_labels[:5]) # [7, 2, 1, 0, 4]
