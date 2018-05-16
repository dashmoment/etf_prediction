import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l

import sklearn.preprocessing as p

tf.reset_default_graph()  
c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(c['src_file_path'])
data = tv_gen._selectData2array(f, ['0050'], None)

train = data[:,:-3]
label = data[:,-3:]

#Only Up and down Features

train_ud = label
ud_feature = train_ud[1:] - train_ud[:-1]
serial_feature = [] 
#[1,0,0]: two day down 
#[0,1,0]: two day fair
#[0,0,1]: two day up

for i in range(1, len(train_ud)):
    
    if train_ud[i][0] == 1 and train_ud[i-1][0] == 1:
        serial_feature.append([1,0,0])
    
    elif train_ud[i][1] == 1 and train_ud[i-1][1] == 1:
        serial_feature.append([0,1,0])
        
    elif train_ud[i][2] == 1 and train_ud[i-1][2] == 1:
        serial_feature.append([0,0,1])
    else:
        serial_feature.append([0,0,0])
        
        
serial_feature_2 = [] 
#[1,0,0]: three day down 
#[0,1,0]: three day fair
#[0,0,1]: three day up

for i in range(2, len(train_ud)):
    
    if train_ud[i][0] == 1 and train_ud[i-1][0] == 1 and train_ud[i-2][0] == 1:
        serial_feature_2.append([1,0,0])
    
    elif train_ud[i][1] == 1 and train_ud[i-1][1] == 1 and train_ud[i-2][1] == 1:
        serial_feature_2.append([0,1,0])
        
    elif train_ud[i][2] == 1 and train_ud[i-1][2] == 1 and train_ud[i-2][2] == 1:
        serial_feature_2.append([0,0,1])
    else:
        serial_feature_2.append([0,0,0])
        
    

serial_feature = np.vstack(serial_feature).astype(np.float64)
serial_feature_2 = np.vstack(serial_feature_2).astype(np.float64)

train_ud = np.hstack([train_ud[2:], ud_feature[1:],serial_feature[1:], serial_feature_2])
label_ud = label[2:]

#train_ud, label_ud = data_label_shift(train_ud, label_ud, lag_day=1)

input_step = 15
predict_step = 5
sample_step = input_step + predict_step

data = []


for i in range(sample_step, len(train_ud)):
    data.append(train_ud[i-sample_step:i])
    
data = np.stack(data, axis=0)

train_data = data[:700]
validation_data = data[750:]

tdata, tlabel = np.split(train_data, [input_step], axis=1)
tlabel = tlabel[:,:,:3]

vdata, vlabel = np.split(validation_data, [input_step], axis=1)
vlabel = vlabel[:,:,:3]


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

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])




