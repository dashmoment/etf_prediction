import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l
import sklearn.preprocessing as prepro
import xgboost as xgb
from sklearn.metrics import accuracy_score

def gather_features(data, feature_mask):

    mask = np.zeros(np.shape(data)[-1], dtype= bool)

    for i in range(len(mask)):
        if i in feature_mask:
            mask[i] = True
        else:
            mask[i] = False

    return data[:,mask]


def get_corr_up_and_down(df, corr_label, mask_index):
    
    corr_matrix = {}
    corr = df.corr()
    corr_matrix['down'] = corr.abs().sort_values(corr.columns[corr_label[0]], ascending=False).drop(corr.index[mask_index])
    corr_matrix['up'] = corr.abs().sort_values(corr.columns[corr_label[1]], ascending=False).drop(corr.index[mask_index])
    
    return corr_matrix


def example_xgb(train, label, mask_features):
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    
    train = gather_features(train, mask_features)
    
    train_xgb = train[:-800]
    label_xgb = np.argmax(label[:-800], -1)
    validation_xgb = train[-800:]
    v_label_xgb = np.argmax(label[-800:], -1) 
    
    model.fit(train_xgb, label_xgb)
    y_xgb_train = model.predict(train_xgb)
    y_xgb_v = model.predict(validation_xgb)
    
    print("Train Accuracy: ", accuracy_score(label_xgb, y_xgb_train))
    print("Validation Accuracy: ",accuracy_score(v_label_xgb, y_xgb_v))
    
    return y_xgb_train, y_xgb_v


def example_xgb_v2(train, label, vtrain, vlabel, mask_features):
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    
    train_xgb = gather_features(train, mask_features)
    validation_xgb = gather_features(vtrain, mask_features)     
    label_xgb = np.argmax(label, -1)
    v_label_xgb = np.argmax(vlabel, -1) 
    
    model.fit(train_xgb, label_xgb)
    y_xgb_train = model.predict(train_xgb)
    y_xgb_v = model.predict(validation_xgb)
    
    print("Train Accuracy: ", accuracy_score(label_xgb, y_xgb_train))
    print("Validation Accuracy: ",accuracy_score(v_label_xgb, y_xgb_v))
    
    return y_xgb_train, y_xgb_v
    
def data_label_shift(train, label, lag_day):
    
    train = train[:-lag_day]
    label = label[lag_day:]
    
    return train, label


def add_DOW(data, axis=1):
    DOW = []
    dow_idx = 1
    for i in range(len(data)):
         
        tmp = [0,0,0,0,0]
        tmp[dow_idx] = 1
        DOW.append(tmp)
        
        if dow_idx > 3:
            dow_idx = 0
        else:
            dow_idx += 1
        
    f_DOW = np.reshape(np.vstack(DOW), (-1,5,1))
    f_DOW = np.repeat(f_DOW, 15, axis=2)
            
    data = np.concatenate((f_DOW, data), axis=axis)    

    return data


def feature_expert_trail(rawdata, feature_mask, label = [], normalize=True, axis=0 ,lagday = 1):

    train = gather_features(rawdata, feature_mask)
    if normalize :  train = prepro.normalize(train, axis=axis)
    if len(label) > 0:
        train_label = label
    else:
        train_label = rawdata[:,-3:]
        
    train, train_label = data_label_shift(train, train_label, lag_day=lagday)

    p = example_xgb(train, train_label, list(range(len(feature_mask))))

    return p

