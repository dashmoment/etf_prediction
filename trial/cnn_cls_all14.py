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

import sklearn.preprocessing as p

tf.reset_default_graph()  
c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()
meta = gu.read_metafile(c['meta_file_path'])
f = tv_gen._load_data(c['src_file_path'])
stock = tv_gen._selectData2array(f, f.index, None)

#******Add Extra Feature*******
stock = add_DOW(stock)


#******************************

stock_diff = stock[1:,:-3] - stock[:-1,:-3]
stock_diff = np.concatenate((stock_diff, stock[1:,-3:]), axis=1)

clean_stock = {}
missin_feature = []
stock_IDs = f.index
feature_names = meta[-1]
for s in range(len(stock_IDs)):
    tmp_stock = stock_diff[:,:,s]
    clean_stock[stock_IDs[s]] = tmp_stock
            
train = []
validation = []
train_raw = {}
validation_raw = {}

for s in stock_IDs:
    
    tmp_train, tmp_validation = tv_gen._split_train_val_side_by_side(clean_stock[s], 20, c['predict_step'], 0.2)
    train.append(tmp_train)
    validation.append(tmp_validation)
    
    train_raw[s] = tmp_train
    validation_raw[s] = tmp_validation
    
train = np.vstack(train)
validation = np.vstack(validation)


train, label_raw = np.split(train, [20], axis=1)
validation, v_label_raw = np.split(validation, [20], axis=1)
feature_mask = list(range(50,80))
tdata =  sesswrapper.gather_features(train, feature_mask)
vdata = sesswrapper.gather_features(validation, feature_mask)
tlabel = label_raw[:,:,-3:]
vlabel = v_label_raw[:,:,-3:]


import keras as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout,Reshape
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt

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






