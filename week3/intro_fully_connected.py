#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%

"""
Created on Sun May 31 13:51:20 2020

@author: marko
"""

'''
Prof. Marko Orescanin

We will work on training a neural network on MNIST dataset. 

# Google explanation 

Training data
We have a dataset of handwritten digits which have been labeled so that we know what each picture represents, i.e. a number between 0 and 9. 

The neural network we will build classifies the handwritten digits in their 10 classes (0, .., 9). 
It does so based on internal parameters that need to have a correct value for the classification to work well. 
This "correct value" is learned through a training process which requires a "labeled dataset" with images and the associated correct answers.

How do we know if the trained neural network performs well or not? Using the training dataset to test the network would be cheating.
It has already seen that dataset multiple times during training and is most certainly very performant on it.
We need another labeled dataset, never seen during training, to evaluate the "real-world" performance of the network. It is called an "validation dataset"

There are 50,000 training digits in this dataset. 
We feed a "batch" of them of size 128 into the training loop at each iteration so the system will have seen all the training digits once after 391 iterations. 
We call this an "epoch".

There are also 10,000 separate "validation" digits for testing the performance of the model.!
'''
# %%
import tensorflow as tf
import os, re, math, json, shutil, pprint
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import IPython.display as display
import numpy as np

from matplotlib import rcParams
#import seaborn as sns
import matplotlib.pyplot as plt #plot
import os


# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
#sns.set()
# sns.set_context('notebook')
# sns.set_context('paper') # smaller
#sns.set_context('talk')  # larger

# change default plot size
rcParams['figure.figsize'] = 8, 6


print("Tensorflow Version: " + tf.__version__ )
print("GPUs available: " + str(len(tf.config.list_physical_devices('GPU'))))
print("CPUs available: " + str(os.cpu_count()))

#print('tensorflow_version:', tf.__version__)
# %%  LOAD DATA
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Lets look at few data samples
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
# %%
rand_14 = np.random.randint(0, x_train.shape[0], 14)
sample_digits = x_train[rand_14]
sample_labels = y_train[rand_14]
# code to view the images
num_rows, num_cols = 2, 7
f, ax = plt.subplots(num_rows, num_cols, figsize=(12, 5),
                     gridspec_kw={'wspace': 0.03, 'hspace': 0.01},
                     squeeze=True)

for r in range(num_rows):
    for c in range(num_cols):
        image_index = r * 7 + c
        ax[r, c].axis("off")
        ax[r, c].imshow(sample_digits[image_index], cmap='gray')
        ax[r, c].set_title('No. %d' % sample_labels[image_index])
plt.show()
plt.close()
#%%
index = 0
print(y_train[index])
print(y_train[index].shape)
# %% make one_hot_encoded
trainY = tf.keras.utils.to_categorical(y_train)
testY = tf.keras.utils.to_categorical(y_test)
index = 0
print(trainY[index])
print(trainY[index].shape)
# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 28*28*1 - > 784

# this is a special loss function, it can work with categorical classes
# generaly you need to provide a one_hot_encoded labels!
# from logits means that it will add softmax activation
# in the process of loss calculation!


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print model layers
model.summary()


#%%  However i dont like sequential models! Functional API is more powerful, checkout 
# keras.io for functional API  28*28 = 784

'''
inputs = tf.keras.Input(shape=(28,28))
flat = tf.keras.layers.Flatten()(inputs)
output = tf.keras.layers.Dense(10, activation='softmax')(flat)
print('inputs shape', inputs.shape)
print('inputs data type', inputs.dtype)

model = tf.keras.Model(inputs=inputs, outputs=output, name="mnist_model")

model.summary()

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
# %%
BATCH_SIZE = 128
EPOCHS = 20

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoint{epoch:02d}-{val_loss:.2f}.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        period=1)
]


history = model.fit(x_train, trainY,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    callbacks=callbacks)

# %%

# plot diagnostic learning curves


def summarize_diagnostics(history):

    plt.figure()
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='val')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.figure()
    plt.title('CrossEntropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='val')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# %%
summarize_diagnostics(history)

# %% now remember we always predict but with the model that did not
# overfit!!

model = tf.keras.models.load_model('checkpoint03-204.42.h5')

# evaluate model
_, acc = model.evaluate(x_test, testY, verbose=0)
print('> %.3f' % (acc * 100.0))

prediction = model.predict(x_train, verbose=0)

# %% LETS ADD Some layers, CASE2

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(60, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])



model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'],
)

model.summary()

"""
You can also plot the model as a graph, you need graphviz package and pydot package and pydotplus package, they are part of my 
requirements, see lectures have correct install sequence:
"""

#tf.keras.utils.plot_model(model, "my_first_model.png")

"""
And, optionally, display the input and output shapes of each layer
in the plotted graph:
"""

#tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

"""
This figure and the code are almost identical. In the code version,
the connection arrows are replaced by the call operation.
A "graph of layers" is an intuitive mental image for a deep learning model,
and the functional API is a way to create models that closely mirror this.
"""
