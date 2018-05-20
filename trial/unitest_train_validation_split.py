import random as rand
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
import seaborn as sns
import sklearn.preprocessing as prepro

#plt.plot(data[:,0,0])

flat_data = np.reshape(np.transpose(data, (0,2,1)), (-1,104))

c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']
*_,meta = gu.read_metafile(c['meta_file_path'])
tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
data = tv_gen._selectData2array(f, f.index, None)


train_sample = flat_data[:-1000]
test_sample = flat_data[-1000:]
ud_mask =  list(range(66, 70)) + list(range(63,66)) + list(range(91,96)) +  list(range(54,62))
train = gather_features(train_sample, ud_mask)
train_label = train_sample[:,-3:]
train, train_label = data_label_shift(train, train_label, lag_day=1)

import xgboost as xgb
from sklearn.metrics import accuracy_score
model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=100, silent=True)
train_xgb = train[:-800]
label_xgb = np.argmax(train_label[:-800], -1)
validation_xgb = train[-800:]
v_label_xgb = np.argmax(train_label[-800:], -1) 
model.fit(train_xgb, label_xgb)
y_xgb_train = model.predict(train_xgb)
y_xgb_v = model.predict(validation_xgb)
print("Train Accuracy: ", accuracy_score(label_xgb, y_xgb_train))
print("Validation Accuracy: ",accuracy_score(v_label_xgb, y_xgb_v))

test = gather_features(test_sample, ud_mask)
test_label = np.argmax(test_sample[:,-3:], axis=-1)
test, test_label = data_label_shift(test, test_label, lag_day=1)
y_xgb_test = model.predict(test)
print("Test Accuracy: ",accuracy_score(test_label, y_xgb_test))

unitest_result = []
for i in range(len(train_sample)):
    for j in range(len(test_sample)):
        
        unitest_result.append(np.equal(train_sample[i], test_sample[j]).all())

unitest_result = any(np.array(unitest_result))