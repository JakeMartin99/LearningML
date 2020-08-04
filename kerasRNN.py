# Sourced from the blog at https://victorzhou.com/blog/keras-rnn-tutorial/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Don't print verbose info msgs
#from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
import numpy

def prepareData(dir):
    data = text_dataset_from_directory(dir)
    return data.map(
        lambda text, label: (regex_replace(text, '<br />', ' '), label),
    )

train_data = text_dataset_from_directory("./aclImdb/train")
test_data = text_dataset_from_directory("./aclImdb/test")

for text_batch, label_batch in train_data.take(1):
    print(text_batch.numpy()[0])
    print(label_batch.numpy()[0]) # 0 = neg, 1 = pos

model = Sequential()
model.add(Input(shape=1,), dtype="string")
