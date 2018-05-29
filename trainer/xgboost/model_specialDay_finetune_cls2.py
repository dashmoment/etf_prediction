import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp
import model_config as mc
import scoreFunc as scoreF

def reduce_label(label):
    
    label_reduce = []
    
    for i in range(len(label)):
        if label[i] == 0 or label[i] == 1: label_reduce.append(0)
        elif label[i] == 2:
            label_reduce.append(1)
            
    return np.array(label_reduce)
            

stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]

stock_list = ['00690']
feature_list_comb_noraml = [
#                                ['velocity'],
#                                ['ma'],
#                                ['ratio'],
#                                ['rsi'],
#                                ['kdj'],
#                                ['macd'],
#                                ['ud'],
                                ['velocity', 'macd'],
                                ['velocity', 'ratio'],
                                ['velocity', 'macd'],
                                ['velocity', 'kdj'],
                            ]

feature_list_comb_special = [                               
#                                ['ratio'],
#                                ['rsi'],
#                                ['kdj'],
#                                ['macd'],
#                                ['ud'],
                                ['ratio', 'macd'],
                                ['ratio', 'macd'],
                                ['ratio', 'kdj'],
                                
                            ]

best_config = {}            
predict_days  = list(range(1, 6))   #The future # day wish model to predict
consider_lagdays = list(range(1,6)) #Contain # lagday information for a training input


model_name = 'xgb_2cls'           
config  = mc.model_config(model_name).get

srcPath = '/home/ubuntu/dataset/etf_prediction/0525/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath =  '/home/ubuntu/dataset/etf_prediction/0525/all_meta_data_Nm_1_MinMax_120.pkl'
corrDate_path = '/home/ubuntu/dataset/etf_prediction/0525/xcorr_date_data.pkl'
#srcPath = '../../Data/0525/all_feature_data_Nm_1_MinMax_120.pkl'
#metaPath = '../../Data/0525/all_meta_data_Nm_1_MinMax_120.pkl'
*_,meta = gu.read_metafile(metaPath)
corrDate = gu.read_datefile(corrDate_path)
corrDate_range = list(range(3,len(corrDate['0050'])+1))  
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

total_progress = len(stock_list)*len(predict_days)*len(consider_lagdays)*len(feature_list_comb_noraml)*len(corrDate_range)
progress = tqdm(total=total_progress)
progress.set_description("[SP][{}]".format(model_name))

for s in stock_list:
    best_config[s] = {}
    progress.set_description("[SP][{}][{}]".format(model_name, s))
    
    for predict_day in predict_days:
        
        best_config[s][predict_day] = {}
        best_accuracy = 0
        if s in ['00690', '00692','00701', '00713']:
            feature_list_comb = feature_list_comb_special
            
        else:
            feature_list_comb = feature_list_comb_noraml
           
        for consider_lagday in consider_lagdays:
            for feature_list in feature_list_comb:
                
                for corr_date in corrDate_range:

                    progress.update(1)
                  
                    #***************Get train data******************
                    single_stock = tv_gen._selectData2array_specialDate(f, corrDate[s][:corr_date], 21, s)      
                    single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
                    features, label = dp.get_data_from_normal(single_stock, meta_v, predict_day, feature_list)
                    
                    label_o  = label
                    label = reduce_label(label)
                    
                    feature_concat = []
                    
                    for i in range(consider_lagday):
                        for k in  features[i]:
                            feature_concat.append(features[i][k])
                    
                    data_feature = np.concatenate(feature_concat, axis=1)
                    train_val_set_days = {'train': data_feature,
                                          'train_label': label}
                    
                
                    train_data = train_val_set_days['train']
                    train_label = train_val_set_days['train_label']
                     
                    
                    #***************Get test data******************
                    single_stock_test = tv_gen._selectData2array(f, [s], ['20180401','20180601'])
                    single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
                    features_test, label_test = dp.get_data_from_normal(single_stock_test, meta_v, predict_day, feature_list)
                    
                    label_test = reduce_label(label_test)
                    
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
                        sample_weight = {'sample_weight':sample_weight}
                        
                    else:
                        sample_weight = {}
                   
                   
                    normal_score = np.mean(cross_val_score(model, train_data, train_label, cv=3,
                                                    n_jobs = 5, 
                                                    #fit_params = sample_weight,
                                                    scoring= scoreF.time_discriminator_score
                                                    ))
                            
                    model.fit(train_data, train_label)
                    y_xgb_train = model.predict(train_data)
                    y_xgb_test = model.predict(test_data)
                    print("Train Accuracy of day {} [SP_cls2][{}]: {}".format(predict_day, model_name, accuracy_score(train_label, y_xgb_train)))
                    print("Validation Accuracy  {} [SP_cls2][{}]: {} ".format(predict_day, model_name, accuracy_score(test_label, y_xgb_test)))
                    
                    score = accuracy_score(y_xgb_test, test_label)
                    #score = normal_score
                    
                    if score >= best_accuracy:    
                        
                        best_accuracy = score
                        gsearch2b = GridSearchCV(model, config['param'], n_jobs=5, cv=3,
                                                scoring= scoreF.time_discriminator_score, 
                                              #fit_params = sample_weight
                                              )
                        gsearch2b.fit(train_data, train_label)
                        fintue_predict = gsearch2b.predict(test_data)
                        fintune_testscore = accuracy_score(test_label, fintue_predict)
                    
                        if fintune_testscore > accuracy_score(test_label, y_xgb_test):
                             best_config[s][predict_day] = {
                                                                    'train acc': accuracy_score(train_label, y_xgb_train),
                                                                    'test_acc': accuracy_score(test_label, y_xgb_test),
                                                                    'days': consider_lagday,
                                                                    'cross_score':normal_score,
                                                                    'features': feature_list,
                                                                    'model_config':gsearch2b.best_params_,
                                                                    'fintune_score': gsearch2b.best_score_,
                                                                    'fintune_testscore': accuracy_score(test_label, fintue_predict),
                                                                    'corrDate':corr_date
                                                                   }
                                
    
                        else:
                              best_config[s][predict_day] = {
                                                                    'train acc': accuracy_score(train_label, y_xgb_train),
                                                                    'test_acc': accuracy_score(test_label, y_xgb_test),
                                                                    'days': consider_lagday,
                                                                    'cross_score':normal_score,
                                                                    'features': feature_list,
                                                                    'model_config':model.get_params(),
                                                                    'fintune_score': gsearch2b.best_score_,
                                                                    'fintune_testscore': accuracy_score(test_label, fintue_predict),
                                                                    'corrDate':corr_date
                                                                   }
  
import pickle
with open('../config/best_config_'+ model_name +'_speicalDate_npw_2cls_cscore.pkl', 'wb') as handle:
    pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    

