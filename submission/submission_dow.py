import sys
sys.path.append('../')

import numpy as np
import model_zoo as mz
import loss_func as l
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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
                                    n_estimators=500,
                                    max_depth = model_config['model_config']['max_depth'],
                                    min_child_weight = model_config['model_config']['min_child_weight'],
                                    objective='multi:softmax', num_class=3)
    def rf(self):
        model_config = self.model_config
        return RandomForestClassifier(n_estimators = 500, max_depth=model_config['model_config']['max_depth'])
    
    def svc(self):
        model_config = self.model_config
        return svm.SVC(
                        C = model_config['model_config']['C'], 
                        gamma=model_config['model_config']['gamma'],
                        kernel=model_config['model_config']['kernel']
                      )

#stock_list =  [
#                '0050', '0051',  '0052', '0053', 
#                '0054', '0055', '0056', '0057', 
#                '0058', '0059', '006201', '006203', 
#                '006204', '006208','00690', '00692',  
#                '00701', '00713'
#              ]



stock_list = ['0050']
predict_days  = list(range(5))
dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}
def get_data_label_pair(single_stock, model_config, meta, isShift=True):
    
    features, label = dp.get_data_from_dow(f, single_stock, meta, predict_day, model_config['features'], isShift)

    feature_concat = []
    for i in range(model_config['days']):
         for k in  features[dow[i]]:
             feature_concat.append( features[dow[i]][k])
    
    data_feature = np.concatenate(feature_concat, axis=1)
    data = data_feature
    label = label
    
    return data, label

srcPath = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl'
tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl')
f = tv_gen._load_data(srcPath)
mConfig =  open('/home/ubuntu/shared/workspace/etf_prediction/trainer/config/best_config_xgb_dow_all_00713.pkl', 'rb')
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
         test_data, test_label = get_data_label_pair(single_stock_test, model_config, meta_v)
         
         model = model_dict('xgb', model_config).get  
         
         model.fit(train_data, train_label)
            
         #********For submission***********
#         test_data = np.reshape(test_data[-1,:], (1,-1))
#         ud = gu.map_ud(model.predict(test_data)[0])
#         predict_ud[s].append(ud)
         
         #********For test************
         p = model.predict(test_data)   
         print(p)
         print(test_label)
         print("Validation Accuracy  {}: {} ".format(predict_day, accuracy_score( test_label, p)))
                    
         
         

    
        
         
#import pickle
#with open('../submission/predict_ud_dow.pkl', 'wb') as handle:
#    pickle.dump(predict_ud, handle, protocol=pickle.HIGHEST_PROTOCOL)       
         
         
         
    
    

