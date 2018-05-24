import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp
import model_config as mc



stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]

stock_list = ['0050']

best_config = {}
date_range = [
#              ['20130601','20150601'],
#              ['20150101','20170601'],
#              ['20160101','20180101'],
              ['20130101','20180401']
             ]
              
predict_days  = list(range(5))  #The dow wish model to predict
consider_lagdays = list(range(1,6)) #Contain # lagday information for a training input
feature_list_comb = [
                        #['velocity'],
                        ['ma'],
                        ['ratio'],
                        ['rsi'],
                        ['kdj'],
                        ['macd'],
                        ['ud']
                    ]
                
config  = mc.model_config('xgb').get

srcPath = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl'
metaPath = '/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl'
#srcPath = '../../Data/all_feature_data_Nm_1_MinMax_94.pkl'
#metaPath = '../../Data/all_meta_data_Nm_1_MinMax_94.pkl'
*_,meta = gu.read_metafile(metaPath)

tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

for s in stock_list:
    best_config[s] = {}
    for predict_day in predict_days:
        
        best_config[s][predict_day] = {}
        best_accuracy = 0
        
        for period in date_range:
            for consider_lagday in consider_lagdays:
                for feature_list in feature_list_comb:
                      
                    #***************Get train data******************
    
                    
                    single_stock = tv_gen._selectData2array(f, [s], period)
                    single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
                    features, label = dp.get_data_from_dow(f , single_stock, meta, predict_day, feature_list)
                    
                    feature_concat = []
                    dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}
                    
                    for i in range(consider_lagday):
                        for k in  features[dow[i]]:
                            feature_concat.append( features[dow[i]][k])          
                    
                    data_feature = np.concatenate(feature_concat, axis=1)
                   
                    train_val_set_days = {'train': data_feature,
                                          'train_label': label}
                    
                
                    train_data = train_val_set_days['train']
                    train_label = train_val_set_days['train_label']
                     
                    
                    #***************Get test data******************
                    single_stock_test = tv_gen._selectData2array(f, [s], ['20180401','20180601'])
                    single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
                    features_test, label_test = dp.get_data_from_dow(f, single_stock_test, meta, predict_day, feature_list)
                    
                    feature_concat_test = []
                    
                    for i in range(consider_lagday):
                        for k in  features_test[dow[i]]:
                            feature_concat_test.append(features_test[dow[i]][k])
                    
                    
                    data_feature_test = np.concatenate(feature_concat_test, axis=1)
                   
                    test_val_set_days = {'test': data_feature_test,
                                          'test_label': label_test}
                    
                    test_data = test_val_set_days['test']
                    test_label = test_val_set_days['test_label']
                    
                    #*************************************************
                    
                    model = config['model']
                    
                    sample_weight = gu.get_sample_weight(train_label)
                    if config['fit_param']:
                        sample_weight = {'sample_weight':sample_weight}
                        
                    else:
                        sample_weight = {}
                    score = np.mean(cross_val_score(model, train_data, train_label, cv=3, 
                                                    fit_params = sample_weight
                                                    ))
                            
                    model.fit(train_data, train_label)
                    y_xgb_train = model.predict(train_data)
                    y_xgb_test = model.predict(test_data)
                    print("Train Accuracy of day {}: {}".format(predict_day, accuracy_score(y_xgb_train, train_label)))
                    print("Validation Accuracy  {}: {} ".format(predict_day, accuracy_score(y_xgb_test, test_label)))
                    
                    if score > best_accuracy:
                        
                         gsearch2b = GridSearchCV(model, config['param'], n_jobs=5, cv=3, fit_params = sample_weight)
                         gsearch2b.fit(train_data, train_label)
                         best_config[s][predict_day] = {'train acc': accuracy_score(y_xgb_train, train_label),
                                                        'test_acc': accuracy_score(y_xgb_test, test_label),
                                                        'days': consider_lagday,
                                                        'cross_score':score,
                                                        'features': feature_list,
                                                        'period':period,
                                                        'model_config':gsearch2b.best_params_,
                                                        'fintune_score': gsearch2b.best_score_}
                         best_accuracy = accuracy_score(y_xgb_test, test_label)
    
   
    
import pickle
with open('../config/best_config_xgb_dow.pkl', 'wb') as handle:
    pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    

