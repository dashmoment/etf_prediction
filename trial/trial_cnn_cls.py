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

train, label_raw = np.split(train, [c['input_step']], axis=1)
validation, v_label_raw = np.split(validation, [c['input_step']], axis=1)

feature_mask = list(range(4))
train = sesswrapper.gather_features(train, feature_mask)
validation = sesswrapper.gather_features(validation, feature_mask)

label = label_raw[:,0,-3:]
label = np.array([label[i,2] == 1 for i in range(len(label))], dtype=np.float32)
v_label = v_label_raw[:,0,-3:]
v_label = np.array([v_label[i,2] == 1 for i in range(len(v_label))], dtype=np.float32)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
enc = OneHotEncoder().fit(np.reshape(label,(-1,1)))

#label = label_raw[:,0,-2:]
#label[:,0] += label_raw[:,0,-3]
#v_label = v_label_raw[:,0,-2:]
#v_label[:,0] += v_label_raw[:,0,-3]
#
##========= add diff features==========
#train_ = np.zeros(np.shape(train))
#validation_ = np.zeros(np.shape(validation))
#
#for i in range(1, len(train[0])):
#    train_[:,i] = train[:,i] - train[:,i-1]
#    
#for i in range(1, len(validation[0])):
#    validation_[:,i] = validation[:,i] - validation[:,i-1]
#    
#train = train_[:,1:]
#validation = validation_[:,1:]



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
model.add(Dense(2))
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
history = model.fit(train, label, epochs=500, batch_size=64,validation_data=(validation, v_label))

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


import xgboost as xgb
from sklearn.metrics import accuracy_score
model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=2000, silent=True)


train_xgb = np.reshape(train, (-1, 120))
validation_xgb = np.reshape(validation, (-1, 120))
label_xgb = label
v_label_xgb = v_label
#label_xgb = np.argmax(label, axis=-1) 
#

model.fit(train_xgb, label_xgb)
y_xgb_train = model.predict(train_xgb)
y_xgb_v = model.predict(validation_xgb)

print(accuracy_score(label_xgb, y_xgb_train))
print(accuracy_score(v_label_xgb, y_xgb_v))





