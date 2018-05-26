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

#stock_list =  [
#                '0050', '0051',  '0052', '0053', 
#                '0054', '0055', '0056', '0057', 
#                '0058', '0059', '006201', '006203', 
#                '006204', '006208','00690', '00692',  
#                '00701', '00713'
#              ]


stock_list = ['0050']
predict_days  = list(range(1,6))
def get_data_label_pair(single_stock, model_config, meta, isShift=True):
    
    features, label = dp.get_data_from_normal(single_stock, meta, predict_day, model_config['features'])

    feature_concat = []
    for i in range(model_config['days']):
         for k in  features[i]:
             feature_concat.append( features[i][k])
    
    data_feature = np.concatenate(feature_concat, axis=1)
    data = data_feature
    label = label
    
    return data, label

srcPath = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl'
tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl')
f = tv_gen._load_data(srcPath)
mConfig =  open('/home/ubuntu/shared/workspace/etf_prediction/trainer/config/best_config_rf_normal.pkl', 'rb')
best_config = pickle.load(mConfig)

predict_ud = {}



for s in stock_list:
     predict_ud[s] = []
     for predict_day in predict_days:
         
         model_config =  best_config[s][predict_day]
         
         single_stock = tv_gen._selectData2array(f, [s], model_config['period'])
         single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
         train_data, train_label = get_data_label_pair(single_stock, model_config, meta_v)
         
         single_stock_test = tv_gen._selectData2array(f, [s], ['20180401', '20180620'])
         single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
         test_data, test_label = get_data_label_pair(single_stock_test, model_config, meta_v, False)
         
#         model = xgb.XGBClassifier( 
#                                   learning_rate= model_config['model_config']['learning_rate'],
#                                   n_estimators=500,
#                                   max_depth = model_config['model_config']['max_depth'],
#                                   min_child_weight = model_config['model_config']['min_child_weight'],
#                                   objective='multi:softmax', num_class=3)
         
         model = RandomForestClassifier(n_estimators = 500, max_depth=model_config['model_config']['max_depth'])
         
         model.fit(train_data, train_label)
            
         #********For submission***********
         test_data = np.reshape(test_data[-1,:], (1,-1))
         ud = gu.map_ud(model.predict(test_data)[0])
         predict_ud[s].append(ud)
         
         #********For test************
#         p = model.predict(test_data)   
#         print(p)
#         print(test_label)
#         print("Validation Accuracy  {}: {} ".format(predict_day, accuracy_score(p, test_label)))
                    
         
         

    
        
         
#import pickle
#with open('../submission/predict_ud_dow.pkl', 'wb') as handle:
#    pickle.dump(predict_ud, handle, protocol=pickle.HIGHEST_PROTOCOL)       
         
         
         
    
    

