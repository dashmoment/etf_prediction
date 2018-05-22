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
import trial_xgboost_ensCls as ens
import xgboost as xgb
from sklearn.metrics import accuracy_score

from utility_trial import *


    

srcPath = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl'
tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl')
f = tv_gen._load_data(srcPath)

dp_  = dp.data_processor(srcPath)



#stock_list =  ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', 
#               '006203', '006204', '006208','00690', '00692', '00701', '00713']

stock_list = ['0050']
period = None
df = pd.DataFrame({'date':f.columns})
df['date'] = pd.to_datetime(df['date'])
df['dow'] = df['date'].dt.dayofweek

#********Get Monday data*********

for s in stock_list:

    single_stock = tv_gen._selectData2array(f, [s], period)
    dow_array = np.array(df['dow'][-len(single_stock):])
    dow_array_mask = np.equal(dow_array, 0)
    monday_stock = single_stock[dow_array_mask]
  
    data = monday_stock[:-1]
    label = np.argmax(monday_stock[1:, -3:], axis=-1)
    
    fe = feature_extractor(meta, data)
    ud, _ = fe.ud()
    data_feature = ud
    train_val_set = dp_.split_train_val_set(data_feature, label, 0.1)
    
    train_fe = ens.feature_extractor(train_val_set['train'], None )
    test_fe = ens.feature_extractor(train_val_set['test'], None)
    train_data_ = train_fe.ratio()
    test_data_ = test_fe.ratio()
    train_label_ = train_val_set['train_label']
    test_label_ = train_val_set['test_label']
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False)
    model.fit(train_data_, train_label_)
    y_xgb_train = model.predict(train_data_)
    y_xgb_valid = model.predict(test_data_)
            
    print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label_))
    print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_valid, test_label_))
    
    
#********Combine two days data*********
    
    
    
    
    
    
    

