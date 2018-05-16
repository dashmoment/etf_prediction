import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
from utility import general_utility as gu
import model_zoo as mz
import loss_func as l
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from utility_trial import *


c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()
meta = gu.read_metafile(c['meta_file_path'])
f = tv_gen._load_data(c['src_file_path'])

data_raw = tv_gen._selectData2array(f, f.index, None)

data_raw = np.transpose(data_raw, (0,2,1))
data_raw = np.reshape(data_raw, (-1,99))

input_step = 20
predict_step = 5
sample_step = input_step + predict_step

data = []
for i in range(sample_step, len(data_raw)):
    data.append(data_raw[i-sample_step:i])

mask = np.zeros((len(data)), bool)
import random

idx = 0
while idx < 2000:
    rnd = random.randint(0,len(mask))
    if not mask[rnd]:
        mask[rnd] = True
        idx += 1
    else: idx = idx

n_mask = np.invert(mask)

data = np.stack(data, axis=0)
train_data = data[1200:]
validation_data = data[:1000]
tdata, tlabel = np.split(train_data, [input_step], axis=1)
tdata = tdata[:,:,50:]
tlabel = tlabel[:,:,-3:]
vdata, vlabel = np.split(validation_data, [input_step], axis=1)
vdata = vdata[:,:,50:]
vlabel = vlabel[:,:,-3:]



#============Build CNN model===============

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout,Reshape
from keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Convolution1D(
        nb_filter = 16,
        kernel_size = 5,
        padding='same',
        input_shape=(np.shape(tdata)[1:])        
))
model.add(Activation('relu'))
model.add(MaxPooling1D(stride=1))
model.add(Convolution1D(
        nb_filter = 16,
        kernel_size = 3,
        padding='same'        
))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(15))
model.add(Reshape((5,3)))
model.add(Activation('softmax'))

model.summary()

softmax_output = Model(inputs=model.input,
                       outputs=model.layers[-1].output)

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
history = model.fit(tdata, tlabel, epochs=500, batch_size=64,validation_data=(vdata, vlabel))

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(vdata, vlabel)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

final_prediction = softmax_output.predict(vdata)
final_prediction_cls = np.argmax(final_prediction, axis=-1)
v_label_cls = np.argmax(vlabel, axis=-1)


test_accuracy = np.mean(np.equal(final_prediction_cls, v_label_cls))
weighted_array = [0.1,0.15,0.2,0.25,0.3]
test_score = np.mean(np.sum(np.equal(final_prediction_cls, v_label_cls).astype(np.float32)*weighted_array*0.5, axis=1))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
