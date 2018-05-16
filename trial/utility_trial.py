import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l

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
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=100, silent=True)
    
    train = gather_features(train, mask_features)
    
    train_xgb = train[800:]
    label_xgb = np.argmax(label[800:], -1)
    validation_xgb = train[:800]
    v_label_xgb = np.argmax(label[:800], -1) 
    
    model.fit(train_xgb, label_xgb)
    y_xgb_train = model.predict(train_xgb)
    y_xgb_v = model.predict(validation_xgb)
    
    print(accuracy_score(label_xgb, y_xgb_train))
    print(accuracy_score(v_label_xgb, y_xgb_v))
    
    return y_xgb_train, y_xgb_v
    
def data_label_shift(train, label, lag_day):
    
    train = train[:-lag_day]
    label = label[lag_day:]
    
    return train, label