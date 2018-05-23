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
import pandas as pd


class feature_extractor:
    def __init__(self, featurelist, data):
        self.featurelist = featurelist
        self.data = data
        
    def rsi(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'RSI' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def kdj(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'KDJ' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def ratio(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'ratio' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def macd(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'MACD' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def ud(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'UD' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask


def get_data_from_dow(raw, stocks, meta, lagfday, feature_list = ['ratio']):
    
    df = pd.DataFrame({'date':raw.columns})
    df['date'] = pd.to_datetime(df['date'])
    df['dow'] = df['date'].dt.dayofweek
    dow_array = np.array(df['dow'][-len(stocks):])
    dow_array_mask_mon =  np.equal(dow_array, lagfday)
     
    def get_mask(dow_array_mask_mon):
         for i in range(5):
             dow_array_mask_mon[i] = False
         
         dow_array_mask = [dow_array_mask_mon]
         for j in range(1, 5):
             tmp_mask = np.zeros(np.shape(dow_array_mask_mon), np.bool)
             for i in range(1, len(dow_array_mask_mon)):
                if dow_array_mask_mon[i] == True: 
                    tmp_mask[i-j] = True              
                else: 
                    tmp_mask[i] = False
             dow_array_mask.append(tmp_mask)
         return dow_array_mask

    dow_array_mask = get_mask(dow_array_mask_mon)
    
    
    dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}
    features = {}
    
    for d in range(5):
        features[dow[d]] = {}
        shifted_stock = stocks[dow_array_mask[d]]
        shifted_stock = shifted_stock[:-1]
        
        fe = feature_extractor(meta, shifted_stock)
        
        for feature_name in feature_list:
            features[dow[d]][feature_name], _ = getattr(fe, feature_name)()
            
    label = np.argmax(stocks[dow_array_mask[0]][1:, -3:], axis=-1)
    
    return features, label

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

def restore_accuracy(predict, label):
    
    y_xgb_predict = []

    for i in range(len(predict)):
    
        if predict[i] == 0: 
            y_xgb_predict.append(0)
        else:
            y_xgb_predict.append(2)

    print("Test Accuracy [ratio]: ", accuracy_score(y_xgb_predict, label))

    return accuracy_score(y_xgb_predict, label)



