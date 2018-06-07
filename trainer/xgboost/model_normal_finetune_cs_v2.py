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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp
import model_config as mc
import scoreFunc as scoreF

stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]

stock_list = ['0050']

date_range_normal = [
                        #['20130101','20150601'],
                        #['20150101','20170101'],
                        ['20130101','20180610'],
                    ]
date_range_special = [['20130101','20180610']]

feature_list_comb_noraml = [
                                ['ma'],
                                ['ratio'],
                                ['macd'],
                                ['kdj'],
                                ['rsi'],
                                ['velocity'],
                                #['ma', 'cont'],
                                ['velocity',  'cont'],               
                                ['ratio',  'cont'],
                                ['rsi',  'cont'],
                                ['kdj',  'cont'],
                                ['macd',  'cont'],
                                #['ud',  'cont'],
                            ]

feature_list_comb_special = [                               
                                ['ratio'],
                                ['macd'],
                                ['kdj'],
                                ['rsi'],
                                ['velocity'],
                                ['velocity',  'cont'],               
                                ['ratio',  'cont'],
                                ['rsi',  'cont'],
                                ['kdj',  'cont'],
                                ['macd',  'cont'],
                                #['ud',  'cont'],
                                
                            ]

              
predict_days  = list(range(1, 6))  #The future # day wish model to predict
consider_lagdays = list(range(1,6)) #Contain # lagday information for a training input

config  = mc.model_config('xgb').get
best_config = {}

srcPath = '/home/ubuntu/dataset/etf_prediction/0525/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath =  '/home/ubuntu/dataset/etf_prediction/0525/all_meta_data_Nm_1_MinMax_120.pkl'
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

        if s in ['00690', '00692','00701', '00713']:
            date_range = date_range_special
            feature_list_comb = feature_list_comb_special
        else:
            date_range = date_range_normal
            feature_list_comb = feature_list_comb_noraml
        
        for period in date_range:
            for consider_lagday in consider_lagdays:
                for feature_list in feature_list_comb:
                      
                      #***************Get train data******************
                    single_stock = tv_gen._selectData2array(f, [s], period)
                    single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
                    single_stock, meta_ud = f_extr.create_ud_cont(single_stock, meta_v)
                    features, label = dp.get_data_from_normal_v2_train(single_stock, meta_ud, predict_day, consider_lagday, feature_list)
                    
                    data_feature = features
                    train_val_set_days = {'train': data_feature,
                                          'train_label': label}                    
                    train_data = train_val_set_days['train']
                    train_label = train_val_set_days['train_label']
                     
                    
                    
                    #*************************************************
                    
                    model = config['model']
                    sample_weight = gu.get_sample_weight(train_label)
                    
                    if config['fit_param']:
                        sample_weight = {'sample_weight':sample_weight}
                        
                    else:
                        sample_weight = {}
                        
                    score = np.mean(cross_val_score(model, train_data, train_label, cv=3, 
                                                    scoring= scoreF.time_discriminator_score,
                                                    #fit_params = sample_weight
                                                    ))
                        
                    model.fit(train_data, train_label)
                    y_xgb_train = model.predict(train_data)
                   
                    print("Train Accuracy of day {} [{}][Noraml]: {}".format(predict_day, s, score))
    
                    
                    if score >= best_accuracy:          
                         best_accuracy = score
                         gsearch2b = GridSearchCV(model, config['param'], n_jobs=5, cv=3,
                                                  scoring= scoreF.time_discriminator_score, 
                                                  #fit_params = sample_weight
                                                  )
                         gsearch2b.fit(train_data, train_label)
            
                         best_config[s][predict_day] = {
                                                        'train acc': accuracy_score(train_label, y_xgb_train),
                                                        'days': consider_lagday,
                                                        'cross_score':score,
                                                        'features': feature_list,
                                                        'period':period,
                                                        'model_config':gsearch2b.best_params_,
                                                        'fintune_score': gsearch2b.best_score_,                                   
                                                        }
                         
    
import pickle
with open('../config/best_config_xgb_normal_onlycs_nsw_v2.pkl', 'wb') as handle:
    pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    

