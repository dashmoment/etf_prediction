import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, classification_report
from fbprophet import Prophet


from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp

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
              ['20130601','20150601'],
              ['20150101','20170601'],
              ['20160101','20180101'],
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
                

#srcPath = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl'
#metaPath = '/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl'
srcPath = '../Data/all_feature_data_Nm_1_MinMax_94.pkl'
metaPath = '../Data/all_meta_data_Nm_1_MinMax_94.pkl'
*_,meta = gu.read_metafile(metaPath)

tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)
single_stock = tv_gen._selectData2array(f, ['0050'], ['20130101', '20170401'])

y = pd.DataFrame({'ds':f.columns[:733], 'y':single_stock[:,3]})

m = Prophet()
m.fit(y)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
m.plot(forecast)

forecast_test  = forecast[733:]
single_stock_test = tv_gen._selectData2array(f, ['0050'], ['20170401', '20170431'])


