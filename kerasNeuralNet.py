import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Don't print verbose info msgs
import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

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

# Build the model
model = Sequential([
    # Layers...
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
print("\nTraining model:")
model.fit(
    train_images, # Training data
    to_categorical(train_labels), # Training targets
    epochs=5,
    batch_size=32,
)

# Test the model
print("\nTesting model:")
model.evaluate(
    test_images,
    to_categorical(test_labels),
)

# Save model to disk
model.save_weights('model.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Predict on the first 5 test images
print("\nPrediction:")
predictions = model.predict(test_images[:5])

# Print the model's predictions
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check predictions against truth
print("Truth:")
print(test_labels[:5]) # [7, 2, 1, 0, 4]
