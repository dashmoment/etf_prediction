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

#testSet = tv_gen._load_data( c['src_file_path'])
#from utility import general_utility as ut
#*_, feature_names = ut.read_metafile(c['meta_file_path'])

train, validation , train_raw, validation_raw, _ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'],c['input_stocks'], 
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        metafile = c['meta_file_path'])

train, label = np.split(train, [c['input_step']], axis=1)
validation, v_label = np.split(validation, [c['input_step']], axis=1)

feature_mask = list(range(4))
train = sesswrapper.gather_features(train, feature_mask)
validation = sesswrapper.gather_features(validation, feature_mask)

#train = np.transpose(train, (0,2,1))
#validation = np.transpose(validation, (0,2,1))
label = label[:,0,-3:]
v_label = v_label[:,0,-3:]


#============Build CNN model===============

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
        input_shape=(np.shape(train)[1:])        
))
model.add(Activation('relu'))
model.add(Convolution1D(
        nb_filter = 64,
        kernel_size = 5,
        padding='same'        
))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
#model.add(Dense(256, activation='relu'))
model.add(Dense(3))
#model.add(Reshape((5,3)))
model.add(Activation('softmax'))

print(model.summary())


softmax_output = Model(inputs=model.input,
                       outputs=model.layers[-1].output)

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
history = model.fit(train, label, epochs=2000, batch_size=64,validation_data=(validation, v_label))

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(validation, v_label)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

final_prediction = softmax_output.predict(validation)
final_prediction_cls = np.argmax(final_prediction, axis=-1)
v_label_cls = np.argmax(v_label, axis=-1)


test_accuracy = np.mean(np.equal(final_prediction_cls, v_label_cls))
weighted_array = [0.1,0.15,0.2,0.25,0.3]
test_score = np.mean(np.sum(np.equal(final_prediction_cls, v_label_cls).astype(np.float32)*weighted_array*0.5, axis=1))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])






