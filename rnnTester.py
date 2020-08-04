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
model.add(LSTM(64, dropout=0.25))
# Dense and output Layers
model.add(Dropout(0.25))
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

# Load model weights
model.load_weights('rnn')
while True:
    for i in range(10):
        val = model.predict([input("Review: ")])[0][0]
        print(str(round(val*100,1)) + '% positive')

    if input("Again(y/n): ") == 'n':
        break
