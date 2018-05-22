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


#for s in stock_list:
#
#    single_stock = tv_gen._selectData2array(f, [s], period)
#    dow_array = np.array(df['dow'][-len(single_stock):])
#    dow_array_mask = np.equal(dow_array, 2)
#    monday_stock = single_stock[dow_array_mask]
#  
#    data = monday_stock[:-1]
#    label = np.argmax(monday_stock[1:, -3:], axis=-1)
#    
#    fe = feature_extractor(meta, data)
#    ud, _ = fe.ratio()
#    data_feature = ud
#    train_val_set = dp_.split_train_val_set(data_feature, label, 0.1)
#    
#    train_data = train_val_set['train']
#    test_data = train_val_set['test']
#    train_label = train_val_set['train_label']
#    test_label = train_val_set['test_label']
#    
#    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False)
#    model.fit(train_data, train_label)
#    y_xgb_train = model.predict(train_data)
#    y_xgb_valid = model.predict(test_data)
#            
#    print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label))
#    print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_valid, test_label))
#    
    
#********Combine two days data*********
    



def get_data_from_dow(stocks, meta, lagfday, feature_list = ['ratio']):
    
    df = pd.DataFrame({'date':f.columns})
    df['date'] = pd.to_datetime(df['date'])
    df['dow'] = df['date'].dt.dayofweek
    dow_array = np.array(df['dow'][-len(single_stock):])
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

stock_list =  ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', 
               '006203', '006204', '006208','00690', '00692', '00701', '00713']


best_config = {}
predict_days  = list(range(5))
consider_lagdays = list(range(1,6))
feature_list_comb = [['ratio'],
                ['rsi'],
                ['kdj'],
                ['macd'],
                ['ud']]

for s in stock_list:
    best_config[s] = {}
    for predict_day in predict_days:
        best_config[s][predict_day] = {}
        best_accuracy = 0
        for consider_lagday in consider_lagdays:
            for feature_list in feature_list_comb:
                  
                
                single_stock = tv_gen._selectData2array(f, [s], period)
                features, label = get_data_from_dow(single_stock, meta, predict_day, feature_list)
                
                feature_concat = []
                dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}
                
                for i in range(consider_lagday):
                    for k in  features[dow[i]]:
                        feature_concat.append( features[dow[i]][k])
                
                
                data_feature = np.concatenate(feature_concat, axis=1)
               
                train_val_set_days = dp_.split_train_val_set(data_feature, label, 0.1)
                
            
                train_data = train_val_set_days['train']
                test_data = train_val_set_days['test']
                train_label = train_val_set_days['train_label']
                test_label = train_val_set_days['test_label']
                
                model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False)
                model.fit(train_data, train_label)
                y_xgb_train = model.predict(train_data)
                y_xgb_valid = model.predict(test_data)
                        
                print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label))
                print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_valid, test_label))
                
                if accuracy_score(y_xgb_valid, test_label) > best_accuracy:
                     best_config[s][predict_day] = {'train acc': accuracy_score(y_xgb_train, train_label),
                                       'test_acc': accuracy_score(y_xgb_valid, test_label),
                                       'days': consider_lagday,
                                       'features': feature_list}
                     best_accuracy = accuracy_score(y_xgb_valid, test_label)
    
   
    
    

    
    
    

