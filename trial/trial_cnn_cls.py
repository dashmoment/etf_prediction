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
train, validation , train_raw, validation_raw, _ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'],c['input_stocks'], 
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        metafile = c['meta_file_path'])
train, label = np.split(train, [50], axis=1)
validation, v_label = np.split(validation, [50], axis=1)
train = np.transpose(train, (0,2,1))
validation = np.transpose(validation, (0,2,1))
label = label[:,:,-3:]
v_label = v_label[:,:,-3:]


#============Build CNN model===============

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout,Reshape
from keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Convolution1D(
        nb_filter = 128,
        kernel_size = 3,
        input_shape=(89, 50)        
))
model.add(Activation('relu'))
model.add(MaxPooling1D())
model.add(Convolution1D(
        nb_filter = 128,
        kernel_size = 3        
))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(15))
model.add(Reshape((5,3)))
model.add(Activation('softmax'))


softmax_output = Model(inputs=model.input,
                       outputs=model.layers[-1].output)

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(train, label, epochs=1000, batch_size=64,validation_data=(validation, v_label))

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(validation, v_label)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

a = softmax_output.predict(validation)


