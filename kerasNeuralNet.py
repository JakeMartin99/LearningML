# Sourced from the blog at https://victorzhou.com/blog/keras-neural-network-tutorial/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Don't print verbose info msgs
import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

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
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.05),
    Dense(256, activation='sigmoid'),
    Dropout(0.15),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.15),
    Dense(32, activation='sigmoid'),
    Dropout(0.05),
    Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
    optimizer=Adam(lr=0.001), # Default is 'adam'
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
print("\nTraining model:")
model.fit(
    train_images, # Training data
    to_categorical(train_labels), # Training targets
    epochs=10,
    batch_size=32,
    validation_data=(test_images, to_categorical(test_labels))
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
