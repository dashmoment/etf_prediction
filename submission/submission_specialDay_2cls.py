import sys
sys.path.append('../')

import numpy as np
import model_zoo as mz
import loss_func as l
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle

from utility import dataProcess as dp
from utility import general_utility as gu
from utility import featureExtractor as f_extr

def load_best_config(configPath):
    
    f = open(configPath, 'rb')
    config = pickle.load(f)
    
    return config

def get_stack_model(stock, days):
    
    configPath = {
                'svc':'/home/ubuntu/shared/workspace/etf_prediction/trainer/config/20180525/best_config_svc_speicalDate.pkl',
                'xgb':'/home/ubuntu/shared/workspace/etf_prediction/trainer/config/20180525/best_config_xgb_speicalDate.pkl',
                'rf':'/home/ubuntu/shared/workspace/etf_prediction/trainer/config/20180525/best_config_rf_speicalDate.pkl'
            }
    
    config = {
                'xgb': load_best_config(configPath['xgb'])[stock][days]['model_config'],
                'rf': load_best_config(configPath['rf'])[stock][days]['model_config'],
                'svc': load_best_config(configPath['svc'])[stock][days]['model_config'],      
            
             }

    return mz.stacking_avg_model(config)


def reduce_label(label):
    
    label_reduce = []
    
    for i in range(len(label)):
        if label[i] == 0: label_reduce.append(0)
        else:
            label_reduce.append(1)
            
    return np.array(label_reduce)

def get_data_label_pair_test(single_stock, model_config, meta, predict_day,isShift=True):
    
    features, label = dp.get_data_from_normal(single_stock, meta, predict_day, model_config['features'], isShift)

    feature_concat = []
    for i in range(model_config['days']):
         for k in  features[i]:
             feature_concat.append( features[i][k])
    
    data_feature = np.concatenate(feature_concat, axis=1)
    data = data_feature
    label = label
    
    return data, label

def get_data_label_pair(f, model_config, meta, predict_day, isShift=True):
    
    single_stock = tv_gen._selectData2array_specialDate_v2(f, corrDate[s][:model_config['corrDate']], model_config['corrDate'], 21, s)
    
    labels = []
    data_feature = []
    for i in range(model_config['corrDate']):
        single_stock_tmp, meta_v = f_extr.create_velocity(single_stock[i], meta)
        single_stock_tmp, meta_ud = f_extr.create_ud_cont_2cls(single_stock_tmp, meta_v)
        features_tmp, label_tmp = dp.get_data_from_normal(single_stock_tmp, meta_ud, predict_day, model_config['features'],isShift)
        label_tmp = reduce_label(label_tmp)
        labels += list(label_tmp)
        
        feature_concat = []
    
        for i in range(model_config['days']):
            for k in  features_tmp[i]:
                feature_concat.append(features_tmp[i][k])
        
        data_feature.append(np.concatenate(feature_concat, axis=1))
        
    data = np.vstack(data_feature)
    label = np.array(labels)
                   
    
    return data, label




stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]


#stock_list = ['00701']
predict_days  = list(range(1,6))



#srcPath = '/home/ubuntu/dataset/etf_prediction/0601/all_feature_data_Nm_1_MinMax_120.pkl'
#metaPath =  '/home/ubuntu/dataset/etf_prediction/0601/all_meta_data_Nm_1_MinMax_120.pkl'
#corrDate_path = '/home/ubuntu/dataset/etf_prediction/0601/xcorr_date_data.pkl'
#mConfig_path = '../trainer/config/best_config_xgb_speicalDate_npw_mfcont_cscore.pkl'

srcPath = '../Data/0608/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath = '../Data/0608/all_meta_data_Nm_1_MinMax_120.pkl'
corrDate_path = '../Data/0608/xcorr_date_data.pkl'
mConfig_path = '../trainer/config/20180608/best_config_xgb_2cls_speicalDate_npw_2cls_cscore.pkl'

tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile(metaPath)
f = tv_gen._load_data(srcPath)
mConfig =  open(mConfig_path, 'rb')
corrDate = gu.read_datefile(corrDate_path)
#corrDate_range = list(range(3,len(corrDate['0050'])+1))  

best_config = pickle.load(mConfig)
predict_ud = {}
predict_ud_rev = {}

class model_dict:
    
    def __init__(self, model, model_config):
        
        self.model_config = model_config
        self.get = self.get_config(model)
            
    def get_config(self, model_name):

        try:
            model = getattr(self, model_name)
            return model()

        except: 
            print("Can not find configuration")
            raise
            
    def xgb(self):
        
        model_config = self.model_config
        return xgb.XGBClassifier( 
                                    learning_rate= model_config['model_config']['learning_rate'],
                                    n_estimators=500,
                                    max_depth = model_config['model_config']['max_depth'],
                                    #min_child_weight = model_config['model_config']['min_child_weight'],
                                    objective='multi:softmax', num_class=3)
        
    def xgb_2cls(self):
        
        model_config = self.model_config
        return xgb.XGBClassifier( 
                                    learning_rate= model_config['model_config']['learning_rate'],
                                    n_estimators=500,
                                    max_depth = model_config['model_config']['max_depth'],
                                    #min_child_weight = model_config['model_config']['min_child_weight'],
                                    objective='multi:softmax', num_class=2)
    def rf(self, model_config):
        model_config = self.model_config
        return RandomForestClassifier(n_estimators = 500, max_depth=model_config['model_config']['max_depth'])
    

mode = 'submission'

if mode == 'submission':
    isShift = False
else:
    isShift = True

for s in stock_list:
     predict_ud[s] = []
     predict_ud_rev[s] = []
     for predict_day in predict_days:
         
         model_config =  best_config[s][predict_day]
         single_stock = tv_gen._selectData2array_specialDate(f, corrDate[s][:model_config['corrDate']], 21, s)
         single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
         single_stock, meta_ud = f_extr.create_ud_cont_2cls(single_stock, meta_v)
         train_data, train_label = get_data_label_pair(f, model_config, meta, predict_day)
         
         single_stock_test = tv_gen._selectData2array(f, [s], ['20180401','20180701'])
         single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
         single_stock_test, meta_ud = f_extr.create_ud_cont_2cls(single_stock_test, meta_v)
         test_data, test_label = get_data_label_pair_test(single_stock_test, model_config, meta_ud, predict_day,isShift)
         test_label = reduce_label(test_label)
         
         
         model = model_dict('xgb_2cls', model_config).get      
         #model = get_stack_model( s, predict_day)    
         model.fit(train_data, train_label)
            
         #********For submission***********
         if mode == 'submission':
             test_data = np.reshape(test_data[-1,:], (1,-1))
             print(test_data.shape, model_config['features'],model_config['days'])
             ud = gu.map_ud_2cls(model.predict(test_data)[0])
             predict_ud[s].append(ud)
             
             ud_rev = gu.map_ud_2cls_reverse(model.predict(test_data)[0])
             predict_ud_rev[s].append(ud_rev)
         
         else:
             #********For test************
             p = model.predict(test_data)   
             print(p)
             print(test_label)
             print("Validation Accuracy  {}: {} ".format(predict_day, accuracy_score(p, test_label)))
                    
         
        
import pickle
with open('./20180608/predict_ud_xgb_speicalDate_nsw_cscore_2cls.pkl', 'wb') as handle:
    pickle.dump(predict_ud, handle, protocol=pickle.HIGHEST_PROTOCOL)  
with open('./20180608/predict_ud_xgb_speicalDate_nsw_cscore_2cls_rev.pkl', 'wb') as handle:
    pickle.dump(predict_ud_rev, handle, protocol=pickle.HIGHEST_PROTOCOL)       
         
         
         
    
    

