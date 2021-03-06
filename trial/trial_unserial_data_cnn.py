import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l

tf.reset_default_graph()  
c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
stock = tv_gen._selectData2array(f, ['2330'], None)

train = stock[:769]
validation = stock[769:]

train_data = np.reshape(train[:,:96], (-1,96,1))
train_label = train[:,96:]
validation_data = np.reshape(validation[:,:96], (-1,96,1))
validation_label = validation[:,96:]

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout,Reshape
from keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Convolution1D(
        nb_filter = 32,
        kernel_size = 5,
        padding='same',
        input_shape= (np.shape(train_data)[1:])       
))
model.add(Activation('relu'))
model.add(MaxPooling1D(stride=1))
model.add(Convolution1D(
        nb_filter = 64,
        kernel_size = 5,
        padding='same'        
))
model.add(Activation('relu'))
#model.add(MaxPooling1D(stride=1))
model.add(Convolution1D(
        nb_filter = 64,
        kernel_size = 5,
        padding='same'        
))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
#model.add(Dense(256, activation='relu'))
model.add(Dense(3))
#model.add(Reshape((5,3)))
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
history = model.fit(train_data, train_label, epochs=500, batch_size=64,validation_data=(validation_data, validation_label))

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(validation_data, validation_label)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

final_prediction = softmax_output.predict(validation)
final_prediction_cls = np.argmax(final_prediction, axis=-1)
v_label_cls = np.argmax(validation_label, axis=-1)


test_accuracy = np.mean(np.equal(final_prediction_cls, v_label_cls))
#weighted_array = [0.1,0.15,0.2,0.25,0.3]
#test_score = np.mean(np.sum(np.equal(final_prediction_cls, v_label_cls).astype(np.float32)*weighted_array*0.5, axis=1))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


