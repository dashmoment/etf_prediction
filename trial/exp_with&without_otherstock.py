#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:00:31 2018

@author: ubuntu
"""

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
import trial_xgboost_ensCls as ens

lagday =  1
tv_gen = dp.train_validation_generaotr()
#*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94_0050.pkl')
#f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94_0050.pkl')
#f = tv_gen._load_data('/home/dashmoment/workspace/etf_prediction/Data/ETF_member/all_feature_data_Nm_1_MinMax_94_0052.pkl')
#*_,meta = gu.read_metafile('/home/dashmoment/workspace/etf_prediction/Data/ETF_member/all_meta_data_Nm_1_MinMax_94_0052.pkl')

stock_list =  ['0050',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', 
               '006203', '006204', '006208','00690', '00692', '00701', '00713']


score = {}
for sk in stock_list:
    
    score[sk] = {}
    print('Scoring stock: ', sk)

    src_path = '/home/ubuntu/dataset/etf_prediction/ETF_member/all_feature_data_Nm_1_MinMax_94_' + str(sk) + '.pkl'
    meta_path = '/home/ubuntu/dataset/etf_prediction/ETF_member/all_meta_data_Nm_1_MinMax_94_' + str(sk) + '.pkl'
    meta = gu.read_metafile(meta_path)
#    _dp = dp.data_processor(src_path, 
#                            lagday = lagday, period=['20130101', '20180311'],
#                            stockList = [sk])
    
    _dp = dp.data_processor(src_path, 
                            lagday = lagday, period=['20130101', '20180311'])
    
    clean_stock = _dp.clean_data()
    train_val_set = _dp.split_train_val_set_mstock(clean_stock, 0.01)
    
    
    train_fe = ens.feature_extractor(train_val_set['train'], None )
    test_fe = ens.feature_extractor(train_val_set['test'], None)
    train_data_ = train_fe.ratio()
    test_data_ = test_fe.ratio()
    
    #train_label_raw = np.stack((train_val_set['train_label'][:, -3] + train_val_set['train_label'][:, -2] , train_val_set['train_label'][:, -1]), axis=1)
    #test_label_raw =  np.stack((train_val_set['test_label'][:, -3] + train_val_set['test_label'][:, -2], train_val_set['test_label'][:, -1]) , axis=1)
    train_label_raw = train_val_set['train_label']
    test_label_raw = train_val_set['test_label']
    train_label_ = np.argmax(train_label_raw, axis=-1)
    test_label_ = np.argmax(test_label_raw, axis=-1)
    test_label_restore = np.argmax(train_val_set['test_label'], axis=-1)
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False)
    model.fit(train_data_, train_label_)
    y_xgb_train = model.predict(train_data_)
    y_xgb_valid = model.predict(test_data_)
            
    print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label_))
    print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_valid, test_label_))
    
    
    score[sk]['train'] = accuracy_score(y_xgb_train, train_label_)
    score[sk]['validation'] = accuracy_score(y_xgb_valid, test_label_)
    score[sk]['validation_restore'] =  restore_accuracy(y_xgb_valid, test_label_restore)
    
    
    
    #*********************************Test by latest data**********************************************
    _dp_latest = dp.data_processor(src_path, 
                            lagday = lagday, period=['20180311', '20180520'])
    
    clean_stock_latest = _dp_latest.clean_data()
    stock = clean_stock_latest[sk]
    test_data_ = stock['data']
    test_label_ = stock['label_ud']
    
    test_fe = ens.feature_extractor(test_data_, None)
    test_data_ = test_fe.ratio()
    #test_label_raw =  np.stack((test_label_[:, -3] + test_label_[:, -2], test_label_[:, -1]) , axis=1)
    test_label_raw = test_label_[:]
    test_label_simple = np.argmax(test_label_raw, axis=-1)
    test_label_restore = np.argmax(test_label_, axis=-1)
    y_xgb_test = model.predict(test_data_)
    
    test_data_5day = test_data_[-5:]
    test_label_5day = test_label_[-5:]
    test_label_simple_5day = np.argmax(test_label_5day, axis=-1)
    test_label_restore_5day = np.argmax(test_label_5day, axis=-1)
    y_xgb_test_5day = model.predict(test_data_5day)
    
    print("Validation Accuracy [ratio]: ", accuracy_score(y_xgb_test_5day, test_label_simple_5day))
    
    
    
    score[sk]['test'] = accuracy_score(y_xgb_test_5day, test_label_simple_5day)
    score[sk]['test_restore'] = restore_accuracy(y_xgb_test_5day, test_label_restore_5day)
    
    
score_all = score 




