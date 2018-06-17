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
                                    n_estimators=model_config['model_config']['n_estimators'],
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
    

stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]

special_stock_list = [
                           '006201', '006203', 
                           '006204', '006208','00690', '00692',  
                           '00701', '00713'        
                     ]


#stock_list = ['0050']
predict_days  = list(range(1,6))

#srcPath = '/home/ubuntu/dataset/etf_prediction/0525/all_feature_data_Nm_1_MinMax_120.pkl'
#metaPath =  '/home/ubuntu/dataset/etf_prediction/0525/all_meta_data_Nm_1_MinMax_120.pkl'
#mConfig_path = '../trainer/config/20180525/best_config_xgb_normal_cv_sc_v2.pkl'
#srcPath_btest = '/home/ubuntu/dataset/etf_prediction/0525/all_feature_data_Nm_1_MinMax_120.pkl'

srcPath = '../Data/0601/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath = '../Data/0601/all_meta_data_Nm_1_MinMax_120.pkl'
srcPath_btest = '../Data/0601/all_feature_data_Nm_1_MinMax_120.pkl'
mConfig_path = '../trainer/config/20180525/best_config_xgb_normal_cv_sc_v2.pkl'


tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile(metaPath)
f = tv_gen._load_data(srcPath)
mConfig =  open(mConfig_path , 'rb')
best_config = pickle.load(mConfig)

predict_ud = {}

submission = True
if submission:
    isShift = False
else:
    isShift = True

for s in stock_list:
     predict_ud[s] = []
     for predict_day in predict_days:
         
         model_config =  best_config[s][predict_day]
         
         single_stock = tv_gen._selectData2array(f, [s],  ['20130101', '20180408'])
         single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
         single_stock, meta_ud = f_extr.create_ud_cont(single_stock, meta_v)
         train_data, train_label = dp.get_data_from_normal_v2_test(single_stock, meta_ud, predict_day, model_config)
         
         single_stock_test = tv_gen._selectData2array(f, [s], ['20180408', '20180610'])
         single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
         single_stock_test, meta_ud = f_extr.create_ud_cont(single_stock_test, meta_v)
         test_data, _ = dp.get_data_from_normal_v2_test(single_stock_test, meta_ud, predict_day, model_config, isShift=isShift)
         #test_data, test_label = get_data_label_pair(single_stock_test, model_config, meta_ud, True)
                     
            
         model = model_dict('xgb', model_config).get 
         model.fit(train_data, train_label)
            
#         #********For submission***********
         
         if submission:
             test_data = np.reshape(test_data[0,:], (1,-1))
             
             if s not in special_stock_list:
             
                 if predict_day == 1 or predict_day==2 or predict_day==4 or predict_day==5:
                     ud = gu.map_ud(2)
                 if predict_day == 3:
                     ud = gu.map_ud(2)
             else:
                if predict_day == 1 or predict_day==2 or predict_day==5 or predict_day==4:
                     ud = gu.map_ud(2)
                #if predict_day == 3 or predict_day==4:
                if predict_day == 3:
                     ud = gu.map_ud(2)
                
            
             predict_ud[s].append(ud)
       
#********Blind test*********  

f_btest = tv_gen._load_data(srcPath_btest)

label_ud_btest = {}

for s in stock_list:
    label_ud_btest[s] = []
    for predict_day in predict_days:
         single_stock_test = tv_gen._selectData2array(f_btest, [s], ['20180408', '20180701'])
         single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
         single_stock_test, meta_ud = f_extr.create_ud_cont(single_stock_test, meta_v)
         _, test_label = dp.get_data_from_normal_v2_test(single_stock_test, meta_ud, predict_day, model_config, isShift=False)
         label_ud_btest[s].append(gu.map_ud(test_label[0]))
     
performance = {}
total_acc = 0
total_wsum = 0
weighting = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
for s in label_ud_btest.keys():
    performance[s] = np.mean(np.array(np.equal(label_ud_btest[s], predict_ud[s]), np.float32))
    performance['w_'+s] = np.sum(weighting*np.array(np.equal(label_ud_btest[s], predict_ud[s]), np.float32))
    total_acc += np.sum(np.array(np.equal(label_ud_btest[s], predict_ud[s]), np.float32))
    total_wsum += performance['w_'+s]
performance['average'] = total_acc/90
performance['w_sum'] = total_wsum*0.5
    

