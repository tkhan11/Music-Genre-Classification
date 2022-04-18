#!/usr/bin/env python
# coding: utf-8
#######
###
##  Author: Tanveer Khan
##
##  Music Genre Classification with Deep Learning
# 
# This tutorial shows how different Convolutional Neural Network architectures are used for the taks of discriminating a piece of audio whether it is music or speech (binary classification).
# 
# The data set used is the [Music Genre](http://marsyasweb.appspot.com/download/data_sets/) data set compiled by George Tzanetakis. It consists of 128 tracks, each 30 seconds long.
#Each class (music/genre) has 64 examples. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

import os
import argparse
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd # Pandas for reading CSV files and easier Data handling in preparation

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from tensorflow.keras.layers import BatchNormalization

# local
#import rp_extract as rp
import audiofile as  audiofile_read

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import __version__ as sklearn_version

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# ## Load the Metadata
# 
# The tab-separated file contains pairs of filename TAB class.
csv_file = './Music_genre/filelist_wclasses.txt' 
metadata = pd.read_csv(csv_file, index_col=0, sep='\t', header=None)
metadata.head(10)

# create list of filenames with associated classes
filelist = metadata.index.tolist()
classes = metadata[1].values.tolist()


# ## Encode Labels to Numbers
# 
# String labels need to be encoded as numbers. We use the LabelEncoder from the scikit-learn package.

print(classes[0:5])

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
labelencoder.fit(classes)
print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))

classes_num = labelencoder.transform(classes)
print(classes_num[0:5])


# Note: In order to correctly re-transform any predicted numbers into strings, we keep the labelencoder for later.
# ## Load the Audio Files

path = './Music_genre/'   ##directory path
list_spectrograms = [] # spectrograms are put into a list first

# desired output parameters
n_mel_bands = 40   # y axis
frames = 80        # x axis

# some FFT parameters
fft_window_size=512
fft_overlap = 0.5
hop_size = int(fft_window_size*(1-fft_overlap))
segment_size = fft_window_size + (frames-1) * hop_size # segment size for desired # frames


for filename in filelist:
    print(".") 
    filepath = os.path.join(path, filename)
    samplerate, samplewidth, wavedata = audiofile_read(filepath,verbose=False)
    sample_length = wavedata.shape[0]

    # make Mono (in case of multiple channels / stereo)
    if wavedata.ndim > 1:
        wavedata = np.mean(wavedata, 1)
        
    # take only a segment; choose start position:
    #pos = 0 # beginning
    pos = wavedata.shape[0]/2 - segment_size/2 # center minus half segment size
    wav_segment = wavedata[pos:pos+segment_size]

    # 1) FFT spectrogram 
    spectrogram = rp.calc_spectrogram(wav_segment,fft_window_size,fft_overlap)

    # 2) Transform to perceptual Mel scale (uses librosa.filters.mel)
    spectrogram = rp.transform2mel(spectrogram,samplerate,fft_window_size,n_mel_bands)
        
    # 3) Log 10 transform
    spectrogram = np.log10(spectrogram)
    
    list_spectrograms.append(spectrogram)
        
print("\nRead", len(filelist), "audio files")

len(list_spectrograms)

spectrogram.shape

print("An audio segment is", round(float(segment_size) / samplerate, 2), "seconds long")


# Note: For simplicity of this tutorial, here we load only 1 single segment of ~ 1 second length from each audio file.
# In a real setting, one would create training instances of as many audio segments as possible to be fed to a Neural Network.
# 
# ### Show Waveform and Spectrogram
# you can skip this if you do not have matplotlib installed

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# show 1 sec wave segment
plt.plot(wav_segment)
filename

# show spectrogram
fig = plt.imshow(spectrogram, origin='lower')
fig.set_cmap('jet')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()


# ## Make 1 big array of list of spectrograms

# a list of many 40x80 spectrograms is made into 1 big array
# config.floatX is from Theano configration to enforce float32 precision (needed for GPU computation)
data = np.array(list_spectrograms, dtype=config.floatX)
print(data.shape)


# ## Standardization
# 
# <b>Always standardize</b> the data before feeding it into the Neural Network!
# 
# As in the Car image tutorial we use <b>Zero-mean Unit-variance standardization</b> (also known as Z-score normalization).
# However, this time we use <b>attribute-wise standardization</b>, i.e. each pixel is standardized individually, as opposed to computing a single mean and single standard deviation of all values.
# 
# ('Flat' standardization would also be possible, but we have seen benefits of attribut-wise standardization in our experiments).
# 
# This time, we use the StandardScaler from the scikit-learn package for our purpose.
# As it works typically on vector data, we have to vectorize (i.e. reshape) our matrices first.

# vectorize
N, ydim, xdim = data.shape
data = data.reshape(N, xdim*ydim)
print("Vectorize data shape:",data.shape)

# standardize
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)

# show mean and standard deviation: two vectors with same length as data.shape[1]
scaler.mean_, scaler.scale_


# # Creating Train & Test Set 
# 
# We split the original full data set into two parts: Train Set (75%) and Test Set (25%).
# 
# Here we compare Random Split vs. Stratified Split:
testset_size = 0.25 # % portion of whole data set to keep for testing, i.e. 75% is used for training

# Normal (random) split of data set into 2 parts
# from sklearn.model_selection import train_test_split

train_set, test_set, train_classes, test_classes = train_test_split(data, classes_num, test_size=testset_size, random_state=0)

train_classes

test_classes

# The two classes may be unbalanced
print("Class Counts: Class 0:", sum(train_classes==0), "Class 1:", sum(train_classes))

# better: Stratified Split retains the class balance in both sets
# from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes_num)

for train_index, test_index in splits:
    print("TRAIN INDEX:", train_index)
    print("TEST INDEX:", test_index)
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes_num[train_index]
    test_classes = classes_num[test_index]
# Note: this for loop is only executed once, if n_iter==1 resp. n_splits==1

print(train_set.shape)
print(test_set.shape)
# Note: we will reshape the data later back to matrix form 

print("Class Counts: Class 0:", sum(train_classes==0), "Class 1:", sum(train_classes))



# For greyscale images, we add the number 1 as the depth of the additional dimension of the input shape (for RGB color images, the number of channels is 3).

n_channels = 1 # for grey-scale, 3 for RGB, but usually already present in the data

'''
if keras.backend.image_dim_ordering() == 'th':
    # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
    train_set = train_set.reshape(train_set.shape[0], n_channels, ydim, xdim)
    test_set = test_set.reshape(test_set.shape[0], n_channels, ydim, xdim)
else:
    # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
    train_set = train_set.reshape(train_set.shape[0], ydim, xdim, n_channels)
    test_set = test_set.reshape(test_set.shape[0], ydim, xdim, n_channels)

'''
# Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
train_set = train_set.reshape(train_set.shape[0], ydim, xdim, n_channels)
test_set = test_set.reshape(test_set.shape[0], ydim, xdim, n_channels)

train_set.shape

test_set.shape

# we store the new shape of the images in the 'input_shape' variable.
# take all dimensions except the 0th one (which is the number of images)
input_shape = train_set.shape[1:]  
input_shape


# # Creating Neural Network Models in Keras
# ## Sequential Models
# 
#
# ## Creating a Single Layer and a Two Layer CNN

# Try: (comment/uncomment code in the following code block)
# * 1 Layer
# * 2 Layer
# * more conv_filters
# * Dropout

#np.random.seed(0) # make results repeatable

model = Sequential()

conv_filters = 16   # number of convolution filters (= CNN depth)
#conv_filters = 32   # number of convolution filters (= CNN depth)

# Layer 1
model.add(Convolution2D(conv_filters, 3, 3, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2))) 
#model.add(Dropout(0.25)) 

# Layer 2
#model.add(Convolution2D(conv_filters, 3, 3))
#model.add(MaxPooling2D(pool_size=(2, 2))) 

# After Convolution, we have a 16*x*y matrix output
# In order to feed this to a Full(Dense) layer, we need to flatten all data
# Note: Keras does automatic shape inference, i.e. it knows how many (flat) input units the next layer will need,
# so no parameter is needed for the Flatten() layer.
model.add(Flatten()) 

# Full layer
model.add(Dense(256, activation='sigmoid')) 

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(1,activation='sigmoid'))


# If you get OverflowError: Range exceeds valid bounds in the above box, check the correct Theano vs. Tensorflow ordering in the box before and your keras.json configuration file.

model.summary()


# ## Training the CNN
# Define a loss function 
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems

# Optimizer = Stochastic Gradient Descent
optimizer = 'sgd' 

# Compiling the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# TRAINING the model
epochs = 15
history = model.fit(train_set, train_classes, batch_size=32, nb_epoch=epochs)


# #### Accuracy goes up pretty quickly for 1 layer on Train set! Also on Test set?

# ### Verifying Accuracy on Test Set

# always execute this, and then a box of accuracy_score below to print the result
test_pred = model.predict_classes(test_set)


# 1 layer
accuracy_score(test_classes, test_pred)

# 2 layer
accuracy_score(test_classes, test_pred)

# 2 layer + 32 convolution filters
accuracy_score(test_classes, test_pred)

# 2 layer + 32 convolution filters + Dropout
accuracy_score(test_classes, test_pred)


# ## Additional Parameters & Techniques
# 
# Try out more parameters and techniques: (comment/uncomment code blocks below)
# * Adding ReLU activation
# * Adding Batch normalization
# * Adding Dropout

model = Sequential()
conv_filters = 16   # number of convolution filters (= CNN depth)

# Layer 1
model.add(Convolution2D(conv_filters, 3, 3, border_mode='valid', input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
#model.add(Dropout(0.3))

# Layer 2
model.add(Convolution2D(conv_filters, 3, 3, border_mode='valid', input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
#model.add(Dropout(0.1))

# In order to feed this to a Full(Dense) layer, we need to flatten all data
model.add(Flatten()) 

# Full layer
model.add(Dense(256))  
#model.add(Activation('relu'))
#model.add(Dropout(0.1))

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(1,activation='sigmoid'))

# Compiling and training the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

epochs = 15
history = model.fit(train_set, train_classes, batch_size=32, nb_epoch=epochs)

# Verifying Accuracy on Test Set

test_pred = model.predict_classes(test_set)
print("Sequential CNN's Accuracy is:",accuracy_score(test_classes, test_pred))


# ## Parallel CNNs
# 
# It has been discovered, that CNNs for music work best, when they have one filter that is detecting frequencies in the vertical axis, and nother filter that is focused on the time axis, i.e. detecting rhythm. Consequently, this is realized in a parallel CNN, where 2 layers are not stacked after each other, but first run independently in parallel with their output being merged later.
# 
# To create parallel CNNs we need a "graph-based" model. In Keras 1.x this is realized via the functional API of the Model() class.
# We use it to create two CNN layers that run in parallel to each other and are merged subsequently.
# In the functional API, you pass the name of the previous layer in (brackets) after defining the next layer.

# Input only specifies the input shape
input = Input(input_shape)

# CNN layers
# specify desired number of filters
n_filters = 16 
# The functional API allows to specify the predecessor in (brackets) after the new Layer function call
conv_layer1 = Convolution2D(n_filters, 10, 2)(input)  # a vertical filter
conv_layer2 = Convolution2D(n_filters, 2, 10)(input)  # a horizontal filter

# possibly add Activation('relu') here

# Pooling layers
maxpool1 = MaxPooling2D(pool_size=(1,2))(conv_layer1) # horizontal pooling
maxpool2 = MaxPooling2D(pool_size=(2,1))(conv_layer2) # vertical pooling

# we have to flatten the Pooling output in order to be concatenated
poolflat1 = Flatten()(maxpool1)
poolflat2 = Flatten()(maxpool2)

# Merge the 2
merged = merge([poolflat1, poolflat2], mode='concat')

full = Dense(256, activation='relu')(merged)
output_layer = Dense(1, activation='sigmoid')(full)

# finally create the model
model = Model(input=input, output=output_layer)

model.summary()

# Define a loss function 
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems

# Optimizer = Stochastic Gradient Descent
optimizer = 'sgd' 

# Compiling the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# TRAINING the model
epochs = 15
history = model.fit(train_set, train_classes, batch_size=32, nb_epoch=epochs)


# ### Verifying Accuracy on Test Set
# 
# Note: The functional API, i.e. Model() does not have a convenience method `.predict_classes()`. We therefore do 'raw' predictions with `predict()`, which returns values between 0 and 1, and then round to the nearest value (0 or 1).

test_pred = model.predict(test_set)
test_pred[0:35,0]

test_pred = np.round(test_pred)
print("Parallel CNN's Accuracy is:", accuracy_score(test_classes, test_pred)


# # Recurrent Neural Network: LSTM
# 
# For an RNN we define the number of time steps to be considered in a sequence. Here, we take a sequence of 10 audio samples from each audio file. The data has to be aligned accordingly to reflect the time steps, we do that in the reshape step below

timesteps = 10
list_spectrograms = [] # spectrograms are put into a list first


# We read all audio files again, as before we kept only 1 segment and now we process 10.

for filename in filelist:
    print ".", 
    filepath = os.path.join(path, filename)
    samplerate, samplewidth, wavedata = audiofile_read(filepath,verbose=False)
    sample_length = wavedata.shape[0]

    # make Mono (in case of multiple channels / stereo)
    if wavedata.ndim > 1:
        wavedata = np.mean(wavedata, 1)
        
    # HERE WE TAKE MULTIPLE SEGMENTS NOW
    pos = 0 # start position of segment
    
    for t in range(timesteps):
        wav_segment = wavedata[pos:pos+segment_size]

        # 1) FFT spectrogram 
        spectrogram = rp.calc_spectrogram(wav_segment,fft_window_size,fft_overlap)

        # 2) Transform to perceptual Mel scale (uses librosa.filters.mel)
        spectrogram = rp.transform2mel(spectrogram,samplerate,fft_window_size,n_mel_bands)

        # 3) Log 10 transform
        spectrogram = np.log10(spectrogram)

        # add to lists
        list_spectrograms.append(spectrogram)
        
         # jump forward exactly by 1 segment
        pos += segment_size   # TODO check if not surpassing file length
        
print("\nRead", len(filelist), "audio files")

print(len(list_spectrograms))

print(len(classes_num_timesteps))


# ### Data Preprocessing (as before)

# make 1 big array
data = np.array(list_spectrograms, dtype=config.floatX)
data.shape

# vectorize
N, ydim, xdim = data.shape
data = data.reshape(N, xdim*ydim)
data.shape

# standardize
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)


# ### Reshaping for RNN
# 
# Now, the second dimension ist not the channel (as in a CNN), but the number of time steps in a sequence to be processed.
# 
N = data.shape[0]/timesteps  # we had N instances (* timesteps)
data = data.reshape(N, timesteps, -1)   # -1 means take the remaining dimensions from the data size given
data.shape

input_dim = data.shape[2]
input_dim

# alternative to input dim
input_shape = data.shape[1:]
input_shape


# ### Create Train/Test Set
# 
# We kept the train_index and test_index from above, and essentially split identically as before:

train_set = data[train_index]
test_set = data[test_index]
train_classes = classes_num[train_index]
test_classes = classes_num[test_index]


# ### Create LSTM Model
from keras.layers import LSTM

model = Sequential()

# TODO add 2-layer CNN before RNN
# conv_filters = 32   # number of convolution filters (= CNN depth)

# Layer 1
#model.add(Convolution2D(conv_filters, 3, 3, input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2))) 
#model.add(Dropout(0.25)) 

# Layer 2
#model.add(Convolution2D(conv_filters, 3, 3))
#model.add(MaxPooling2D(pool_size=(2, 2))) 

# In order to feed this to a Full(Dense) layer, we need to flatten all data
#model.add(Flatten()) 

# LSTM layer with 256 units
model.add(LSTM(output_dim=256, input_length=timesteps, input_dim=input_dim, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))

# Output layer with 1 unit
model.add(Dense(1))
model.add(Activation('sigmoid'))


# for RNNs take RMSprop as the optimizer!
optimizer='rmsprop'

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = 15
validation_data = (test_set, test_classes)

history = model.fit(train_set, train_classes, batch_size=16, nb_epoch=epochs, validation_data=validation_data)


# ### Verifying Test Set Accuracy

test_pred = model.predict_classes(test_set)
print("LSTM Accuracy is:", accuracy_score(test_classes, test_pred)

# alternative
score = model.evaluate(test_set, test_classes, batch_size=16)
score
