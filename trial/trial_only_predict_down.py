import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def gather_features(data, feature_mask):

    mask = np.zeros(np.shape(data)[-1], dtype= bool)

    for i in range(len(mask)):
        if i in feature_mask:
            mask[i] = True
        else:
            mask[i] = False

    return data[:,mask]

tf.reset_default_graph()  
c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
stock = tv_gen._selectData2array(f, ['0050'], None)

train = stock[:769]
validation = stock[769:]

train_data = train
train_label = train[:,-3:]
validation_data = validation
validation_label = validation[:,-3:]

#Train shift 1 day
lag_day = 5
train_data = train_data[:-lag_day]
train_label = train_label[lag_day:]
validation_data = validation_data[:-lag_day]
validation_label = validation_label[lag_day:]


total_info = np.hstack((train_data, train_label))
total_info = pd.DataFrame(total_info)

corr = total_info.corr()
mask_index = [86,87,88,89,90,91]
corr_sort_down = corr.abs().sort_values(corr.columns[89], ascending=False).drop(corr.index[mask_index])
corr_sort_up = corr.abs().sort_values(corr.columns[91], ascending=False).drop(corr.index[mask_index])


train_df = pd.DataFrame(train)

corr = train_df.corr()
def get_label_down(train_label):
    
    train_label_down = []

    for i in range(len(train_label)):
        
        if train_label[i,0] == 1:
            train_label_down.append(1)
        else:
            train_label_down.append(0)
    
    train_label_down = np.vstack(train_label_down)
    return train_label_down

feature_mask = [71, 4, 63, 64]

train_down = gather_features(train_data, feature_mask)
validation_down = gather_features(validation_data, feature_mask)

train_label_down = np.squeeze(np.argmax(train_label, axis=-1))
validation_label_down = np.squeeze(np.argmax(validation_label, axis=-1))
#train_label_down = get_label_down(train_label)
#validation_label_down = get_label_down(validation_label)


import xgboost as xgb
from sklearn.metrics import accuracy_score
model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=100, silent=True)


train_xgb = train_down
validation_xgb = validation_down
label_xgb = train_label_down
v_label_xgb = validation_label_down
#label_xgb = np.argmax(label, axis=-1) 
#

model.fit(train_xgb, label_xgb)
y_xgb_train = model.predict(train_xgb)
y_xgb_v = model.predict(validation_xgb)

print(accuracy_score(label_xgb, y_xgb_train))
print(accuracy_score(v_label_xgb, y_xgb_v))

#result = []
#for i in range(len(y_xgb_v)):
#    if y_xgb_v[i] > 0:
#        result.append(0)
#    else:
#        result.append(2)
#        
#result = np.array(result)
#validation_label_cls = np.argmax(validation_label, axis=-1)
#
#
#acc = np.mean(np.equal(result, validation_label_cls).astype(np.float32))
#
#print(accuracy_score(validation_label_cls, result))







