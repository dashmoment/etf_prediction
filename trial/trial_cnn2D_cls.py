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
import random as rand
from utility_trial import *

import sklearn.preprocessing as p

c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()
meta = gu.read_metafile(c['meta_file_path'])
f = tv_gen._load_data(c['src_file_path'])
stock = tv_gen._selectData2array(f, f.index, None) 

stock = add_DOW(stock, axis=1) 


train = stock[:-200]
validation = stock[-200:]

train_t = []
validation_t = []

for i in range(0, len(train) - sample_window, 1):    
    train_t.append(train[i:i+sample_window])
train_t = np.stack(train_t)
    
for i in range(0, len(validation) - sample_window, 1):    
    validation_t.append(validation[i:i+sample_window])
validation_t = np.stack(validation_t)  

#t_sample = []
#for i in range(5000):
#    
#    n = rand.randint(0,len(train_t)-1)
#    i1 = rand.randint(0,14)
#    i2 = rand.randint(0,14)
#    i3  = rand.randint(0,14)
#    
#    t_sample.append(np.stack([train_t[n,:,:,i1],train_t[n,:,:,i2],train_t[n,:,:,i3]], axis=2))
#
#t_sample = np.stack(t_sample)
#
#v_sample = []
#for i in range(200):
#    
#    n = rand.randint(0,len(validation_t)-1)
#    i1 = rand.randint(0,14)
#    i2  = rand.randint(0,14)
#    i3 = rand.randint(0,14)
#    
#    v_sample.append(np.stack([validation_t[n,:,:,i1],validation_t[n,:,:,i2],validation_t[n,:,:,i3]], axis=2))
#
#v_sample = np.stack(v_sample)
t_sample = train_t
v_sample = validation_t
train_t, label_t = np.split(t_sample, [c['input_step']], axis=1)
#train_t = train_t[:,:,50:69]
label_t = np.transpose(label_t[:,:,-3:, :], (0,3,1,2))


validation_t, vlabel_t = np.split(v_sample, [c['input_step']], axis=1)
#validation_t = validation_t[:,:,50:69]
vlabel_t = np.transpose(vlabel_t[:,:,-3:, :], (0,3,1,2))

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout,Reshape
from keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model = Sequential()
model.add(Convolution2D(
        nb_filter = 16,
        kernel_size = 3,
        padding='same',
        input_shape=(np.shape(train_t)[1:])        
))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Convolution2D(
        nb_filter = 16,
        kernel_size = 3,
        padding='same',
        input_shape=(np.shape(train_t)[1:])        
))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(225, activation='relu'))
model.add(Reshape((15,5,3)))
model.add(Activation('softmax'))

model.summary()

softmax_output = Model(inputs=model.input,
                       outputs=model.layers[-1].output)
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy', metrics=['accuracy'])


print('Training ------------')
# Another way to train the model
history_ = model.fit(train_t, label_t, epochs=1000, batch_size=64,validation_data=(validation_t, vlabel_t))

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(validation_t, vlabel_t)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

final_prediction = softmax_output.predict(validation_t)
final_prediction_cls = np.argmax(final_prediction, axis=-1)
v_label_cls = np.argmax(vlabel_t, axis=-1)
test_accuracy = np.mean(np.equal(final_prediction_cls, v_label_cls))





 

