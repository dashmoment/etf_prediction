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

#stock_list = ['0050']

best_config = {}            
predict_days  = list(range(1, 6))   #The future # day wish model to predict
consider_lagdays = list(range(1,6)) #Contain # lagday information for a training input
feature_list_comb = [
                        ['ratio'],
                        ['rsi'],
                        ['kdj'],
                        ['macd'],
                        ['ud']
                    ]

model_config = mc.model_config()                 
config = model_config['xgb']

srcPath = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl'
metaPath = '/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl'
*_,meta = gu.read_metafile(metaPath)
corrDate = gu.read_datefile('/home/ubuntu/dataset/etf_prediction/corr_date/xcorr_date_data.pkl')
corrDate_range = list(range(3,len(corrDate['0050'])+1))  
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)


for s in stock_list:
    best_config[s] = {}
    for predict_day in predict_days:
        
        best_config[s][predict_day] = {}
        best_accuracy = 0
        
        
        for consider_lagday in consider_lagdays:
            for feature_list in feature_list_comb:
                
                for corr_date in corrDate_range:
                  
                     #***************Get train data******************
                    single_stock = tv_gen._selectData2array_specialDate(f, corrDate[s][:corr_date], 21, s)
                    features, label = dp.get_data_from_normal(single_stock, meta, predict_day, feature_list)
                    
                    feature_concat = []
                    
                    for i in range(consider_lagday):
                        for k in  features[i]:
                            feature_concat.append( features[i][k])
                    
                    data_feature = np.concatenate(feature_concat, axis=1)
                    train_val_set_days = {'train': data_feature,
                                          'train_label': label}
                    
                
                    train_data = train_val_set_days['train']
                    train_label = train_val_set_days['train_label']
                     
                    
                    #***************Get test data******************
                    single_stock_test = tv_gen._selectData2array(f, [s], ['20180401','20180601'])
                    features_test, label_test = dp.get_data_from_normal(single_stock_test, meta, predict_day, feature_list)
                    
                    feature_concat_test = []
                    
                    for i in range(consider_lagday):
                        for k in  features_test[i]:
                            feature_concat_test.append(features_test[i][k])
                    
                    
                    data_feature_test = np.concatenate(feature_concat_test, axis=1)
                   
                    test_val_set_days = {'test': data_feature_test,
                                          'test_label': label_test}
                    
                    test_data = test_val_set_days['test']
                    test_label = test_val_set_days['test_label']
                    
                    #*************************************************
                    
                    model = config['model']
                    
                    sample_weight = gu.get_sample_weight(train_label)
                    if config['fit_param']:
                        sample_weight = sample_weight
                        
                    else:
                        sample_weight = {}
                    score = np.mean(cross_val_score(model, train_data, train_label, cv=3, 
                                                    fit_params = {'sample_weight':sample_weight}))
                   
                    model.fit(train_data, train_label)
                    y_xgb_train = model.predict(train_data)
                    y_xgb_test = model.predict(test_data)
                    print("Train Accuracy of day {}: {}".format(predict_day, accuracy_score(y_xgb_train, train_label)))
                    print("Validation Accuracy  {}: {} ".format(predict_day, accuracy_score(y_xgb_test, test_label)))
                    
                    if score > best_accuracy:
                        
                         gsearch2b = GridSearchCV(model,  config['param'], n_jobs=5, cv=3, fit_params = {'sample_weight':sample_weight})
                         gsearch2b.fit(train_data, train_label)
                         best_config[s][predict_day] = {'train acc': accuracy_score(y_xgb_train, train_label),
                                                        'test_acc': accuracy_score(y_xgb_test, test_label),
                                                        'days': consider_lagday,
                                                        'cross_score':score,
                                                        'features': feature_list,
                                                        'model_config':gsearch2b.best_params_,
                                                        'fintune_score': gsearch2b.best_score_,
                                                        'corrDate':corr_date}
                         best_accuracy = accuracy_score(y_xgb_test, test_label)
  
import pickle
with open('../config/best_config_speicalDate.pkl', 'wb') as handle:
    pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    

