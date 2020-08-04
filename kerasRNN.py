# Sourced from the blog at https://victorzhou.com/blog/keras-rnn-tutorial/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Don't print verbose info msgs
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy

def prepareData(dir):
    data = text_dataset_from_directory(dir)
    return data.map(
        lambda text, label: (regex_replace(text, '<br />', ' '), label),
    )

train_data = prepareData("./aclImdb/train")
test_data = prepareData("./aclImdb/test")

# Create sequential NN model, with string input layer
model = Sequential()
model.add(Input(shape=(1,), dtype="string"))

# Convert string input into sequence of token ints
max_tokens = 1000
max_len = 100
vectorize_layer = TextVectorization(
    # Max vocab size for tokens consider in-vocab
    max_tokens=max_tokens,
    # Output integer indices, one per string token
    output_mode="int",
    # Pad/truncate to always be exact length of tokens
    output_sequence_length=max_len,
)
train_texts = train_data.map(lambda text, label: text)
vectorize_layer.adapt(train_texts)
model.add(vectorize_layer)

# Convert ints to fixed-len vectors, vocab + 1 for out-of-vocab
model.add(Embedding(max_tokens+1, 128))
# Recurrent LSTM layers with n-size output space
model.add(LSTM(64))
model.add(LSTM(64))
# Dense and output Layers
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Train the model
print("\nTraining model:")
model.fit(train_data, epochs=10, validation_data=test_data)

# Save model weights
model.save_weights('rnn')

# Load model weights and evaluate on test data
model.load_weights('rnn')
model.evaluate(test_data)


# Should print a very high score like 0.98.
print(model.predict([
  "i loved it! highly recommend it to anyone and everyone looking for a great movie to watch.",
]))

# Should print a very low score like 0.01.
print(model.predict([
  "this was awful! i hated it so much, nobody should watch this. the acting was terrible, the music was terrible, overall it was just bad.",
]))
