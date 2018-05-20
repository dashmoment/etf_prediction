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
import sklearn.preprocessing as prepro
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score

from utility_trial import *

class feature_extractor:
    
    def __init__(self, flatData, velocity):
        self.flatData = flatData    
        self.velocity = velocity
        
    def ratio(self):        
        feature_mask = list(range(81,86)) 
        sample =  gather_features(self.flatData, feature_mask)
        
        return sample
    
    def kdj_ratio(self):      
        kgj_ratio_mask =  list(range(66, 70)) + list(range(81,86))
        sample =  gather_features(self.flatData, kgj_ratio_mask)
        return sample
    
    def ratio_velocity(self):       
        sample_ratio =  self.ratio()
        sample_r_v = np.concatenate([self.velocity, sample_ratio], axis=1)
        return sample_r_v
    
    def ud(self):
        ud_mask =  list(range(91, 94))
        sample =  gather_features(self.flatData, ud_mask)
        return sample
        
    def kdj_macd_rssi_ratio(self):
        kmrr_mask =  list(range(66, 70)) + list(range(63,66)) + list(range(81,86)) +  list(range(54,62))
        sample =  gather_features(self.flatData, kmrr_mask)
        return sample


def get_ens_model(lagday = 5, model_temp = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)):
    
    print('**********Generate model for {} day***********'.format(lagday))
    
    c = conf.config('trial_cnn_cls').config['common']
    *_,meta = gu.read_metafile(c['meta_file_path'])
    tv_gen = dp.train_validation_generaotr()  
    f = tv_gen._load_data(c['src_file_path'])
    data = tv_gen._selectData2array(f, f.index[:-4], None)
    
    data_velocity= (data[1:,0:4] - data[:-1,0:4])/(data[:-1,0:4] + 0.1)
    data = data[1:]
    
    train_sample = data[:-30]
    train_sample_v = data_velocity[:-30]
    flat_train_sample = np.reshape(np.transpose(train_sample, (0,2,1)), (-1,94))
    flat_train_sample_velocity = np.reshape(np.transpose(train_sample_v, (0,2,1)), (-1,4))
    
    test_sample = data[-30:]
    test_sample_v = data_velocity[-30:]
    flat_test_sample = np.reshape(np.transpose(test_sample, (0,2,1)), (-1,94))
    flat_test_sample_velocity = np.reshape(np.transpose(test_sample_v, (0,2,1)), (-1,4))
    
#    
#    flat_train_sample = train_data['train']
#    flat_train_sample_velocity = train_data['train_velocity']
#    
#    flat_test_sample = test_data['test']
#    flat_test_sample_velocity = test_data['test_velocity']
    
    
    fe_train = feature_extractor(flat_train_sample, flat_train_sample_velocity)
    d_ratio = fe_train.ratio()
    d_kdj_ratio = fe_train.kdj_ratio()
    d_ratio_velocity = fe_train.ratio_velocity()
    d_ud = fe_train.ud()
    d_kdj_macd_rssi_ratio = fe_train.kdj_macd_rssi_ratio()
    
    fe_test = feature_extractor(flat_test_sample, flat_test_sample_velocity)
    d_ratio_test = fe_test.ratio()
    d_kdj_ratio_test = fe_test.kdj_ratio()
    d_ratio_velocity_test = fe_test.ratio_velocity()
    d_ud_test = fe_test.ud()
    d_kdj_macd_rssi_ratio_test = fe_test.kdj_macd_rssi_ratio()
    
    train_label_raw = np.stack((flat_train_sample[:, -3] + flat_train_sample[:, -2] , flat_train_sample[:, -1]), axis=1)
    test_label_raw =  np.stack((flat_test_sample[:, -3] + flat_test_sample[:, -2], flat_test_sample[:, -1]) , axis=1)
    
    model_dict = {}
    predict_dict = {}
    #*****ratio********
    train, train_label = data_label_shift(d_ratio, train_label_raw, lag_day=lagday)
    test, test_label = data_label_shift(d_ratio_test, test_label_raw, lag_day=lagday)
    train_label = np.argmax(train_label, axis=-1)
    test_label = np.argmax(test_label, axis=-1)
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    model.fit(train, train_label)
    model_dict['ratio'] = model
    
    y_xgb_train = model.predict(train)
    y_xgb_v = model.predict(test)
    predict_dict['ratio'] = [y_xgb_train, y_xgb_v]
        
    print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label))
    print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_v, test_label))
    
    
    #*****kdj_ratio********
    train =  d_kdj_ratio[:-lagday]
    test = d_kdj_ratio_test[:-lagday]
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    model.fit(train, train_label)
    model_dict['kdj_ratio'] = model
    
    y_xgb_train = model.predict(train)
    y_xgb_v = model.predict(test)
    predict_dict['kdj_ratio'] = [y_xgb_train, y_xgb_v]
        
    print("Train Accuracy [kdj_ratio]: ", accuracy_score(y_xgb_train, train_label))
    print("Validation Accuracy [kdj_ratio]: ",accuracy_score(y_xgb_v, test_label))
    
    #*****ratio_velocity********
    train =  d_ratio_velocity[:-lagday]
    test = d_ratio_velocity_test[:-lagday]
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    model.fit(train, train_label)
    model_dict['ratio_velocity'] = model
    
    y_xgb_train = model.predict(train)
    y_xgb_v = model.predict(test)
    predict_dict['ratio_velocity'] = [y_xgb_train, y_xgb_v]
        
    print("Train Accuracy [ratio_velocity]: ", accuracy_score(y_xgb_train, train_label))
    print("Validation Accuracy [ratio_velocity]: ",accuracy_score(y_xgb_v, test_label))
    
    #*****ud********
    train =  d_ud[:-lagday]
    test = d_ud_test[:-lagday]
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    model.fit(train, train_label)
    model_dict['ud'] = model
    
    y_xgb_train = model.predict(train)
    y_xgb_v = model.predict(test)
    
    predict_dict['ud'] = [y_xgb_train, y_xgb_v]
    
    print("Train Accuracy [ud]: ", accuracy_score(y_xgb_train, train_label))
    print("Validation Accuracy [ud]: ",accuracy_score(y_xgb_v, test_label))
    
    
    #*****kdj_macd_rssi_ratio********
    train =  d_kdj_macd_rssi_ratio[:-lagday]
    test = d_kdj_macd_rssi_ratio_test[:-lagday]
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True)
    model.fit(train, train_label)
    model_dict['kdj_macd_rssi_ratio'] = model
    
    y_xgb_train = model.predict(train)
    y_xgb_v = model.predict(test)
    
    predict_dict['kdj_macd_rssi_ratio'] = [y_xgb_train, y_xgb_v]
    
    print("Train Accuracy [kdj_macd_rssi_ratio]: ", accuracy_score(y_xgb_train, train_label))
    print("Validation Accuracy [kdj_macd_rssi_ratio]: ",accuracy_score(y_xgb_v, test_label))
    
    
    #*********Generate assemble input***********
    
    predict_train = []
    predict_test = []
    for k in predict_dict:
        predict_train.append(predict_dict[k][0])
        predict_test.append(predict_dict[k][1])
    
    predict_train = np.stack(predict_train, axis=1)
    predict_test = np.stack(predict_test, axis=1)
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=10, silent=True)
    model.fit(predict_train, train_label)
    model_dict['ensemble'] = model
    y_xgb_train_ens = model.predict(predict_train)
    y_xgb_v_ens = model.predict(predict_test)
    
    print("Train Accuracy [Ens]: ", accuracy_score(y_xgb_train_ens, train_label))
    print("Validation Accuracy [Ens]: ",accuracy_score(y_xgb_v_ens, test_label))
    
    return model_dict



