from __future__ import print_function

import numpy as np
from tensorflow import keras
reuters = keras.datasets.reuters


max_words = 1000
batch_size = 32
epochs = 5

print('Loading data...')
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(train_data), 'train sequences')
print(len(test_data), 'test sequences')


print(train_data[0])

word_index = reuters.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))
