from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd

import numpy as np
import sys
import collections
import itertools

imdb = keras.datasets.imdb

df = pd.read_csv("./fake_or_real_news.csv")
# Set `y`
train_lables = [1 if label == 'REAL' else 0 for label in df.label]
# Drop the `label` column
df.drop("label", axis=1)

train_data = df['text']

#print(train_lables)
#print(train_data)


# Pre Proccesing
count = [['<PAD>', 0], ['<START>', 1], ["<UNK>", 2]]

words1 = ['the', 'quick', 'brown', 'jumped', 'over', 'the', 'lazy', 'dog']

#words = list(itertools.chain.from_iterable(train_data))
print(train_data)

'''
count.extend(collections.Counter(words).most_common(7 - 1))
dictionary = dict()

for word, _ in count:
        dictionary[word] = len(dictionary)
data = list()
unk_count = 0
for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0  # dictionary['UNK']
        unk_count += 1
    data.append(index)
count[0][1] = unk_count
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

print(dictionary)




(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
word_index = dict([(value, key) for (value, key) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def encode_review(text):
    return ([word_index.get(i, '?') for i in text.split()])


#for i in train_data[0]:
#    print(' '.join([reverse_word_index.get(i, '?')]))

#print(decode_review(['<START>', '<PAD>', '<PAD>', '<PAD>']))

user_review = []
user_review.append(encode_review('this movie is good'))
print(user_review[0])

print(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
                                                       
user_review = keras.preprocessing.sequence.pad_sequences(user_review,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

def predict(value, index):
    predictions = model.predict(value)
    if predictions[index] > 0.6:
        return 'Positve ' + str(predictions[index])
    elif predictions[index] < 0.4:
        return 'Negative ' + str(predictions[index])
    else:
        return 'idk ' + str(predictions[index])

print(predict(user_review, 0))
'''