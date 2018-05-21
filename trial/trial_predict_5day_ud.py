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
import sklearn.preprocessing as prepro
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score

from utility_trial import *

tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl')
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl')

clean_stocks = {}
for s in f.index:

    single_stock = tv_gen._selectData2array(f, [s], ['20150207','20160118'])
    
    tmpStock = []
    for i in range(len(single_stock)):
        if not np.isnan(single_stock[i,0:5]).all():
            tmpStock.append(single_stock[i])
    clean_stocks[s] = np.array(tmpStock)
    
    stock = clean_stocks['0050']

stocks_lag5 = []
stocks_lag5_label = []

global_prob = np.zeros((5,3)) 
true_series_label = []

inputs = stock[:-5, :-3]
labels = stock[5:, -3:]

shifted_data = np.concatenate((inputs, labels), axis=1)

for i in range(0, len(shifted_data)-5, 2):
    
    tmp_up = 0
    tmp_down = 0
    tmp_fair = 0
    tmp_info = []
    tmp_label = []
    
    for j in range(5):
        if shifted_data[i+j,-3] == 1: 
            tmp_down+=1
            global_prob[j,0] += 1 
        elif shifted_data[i+j,-2] == 1: 
            tmp_fair+=1
            global_prob[j,1] += 1 
        elif shifted_data[i+j,-1] == 1: 
            tmp_up+=1
            global_prob[j,2] += 1 
        
        tmp_info.append(shifted_data[i+j, 66:70])
        tmp_label.append(shifted_data[i+j,-3:])
    
    stocks_lag5.append(np.concatenate(tmp_info))  
    stocks_lag5_label.append([tmp_down, tmp_fair, tmp_up])
    true_series_label.append(tmp_label)

global_prob = global_prob/np.reshape(np.sum(global_prob, axis=1), (-1,1))

#**********Prepare training validation set *************

stocks_lag = np.vstack(stocks_lag5)
stocks_lag_label = np.vstack(stocks_lag5_label).astype(np.float64)


stocks_lag5_train = stocks_lag[:-20]
stocks_lag5_label_train = stocks_lag_label[:-20]

test = stocks_lag[-20:]
test_label = stocks_lag_label[-20:]


import xgboost as xgb
import sklearn.multioutput as sk
basemodel = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1)
model_xgb = sk.MultiOutputRegressor(basemodel)

m = model_xgb.fit(stocks_lag5_train, stocks_lag5_label_train)
p = m.predict(stocks_lag5_train)
p_around  = np.around(p) 
acc = np.mean(np.equal(p_around, stocks_lag5_label_train))
mse = ((p_around - stocks_lag5_label_train)**2).mean()

p_test = m.predict(test)
p_around_test  = np.around(p_test) 
acc_test = np.mean(np.equal(p_around_test, test_label))
mse_test = ((p_around_test - test_label)**2).mean()
    
    
plt.plot(test_label[:,0])
plt.plot(p_test[:,0])

def acc_scorer(estimator, x, y):
    p_test = estimator.predict(x)
    p_around_test  = np.around(p_test) 
    
    acc_test = np.mean(np.equal(p_around_test, y))
    
    return acc_test

n_estimators = list(range(20,100,20))
n_depth = [1,2,3,4,5]

score = {}
best_score = 0
from sklearn.model_selection import cross_val_score
for Nest in n_estimators:
    for d in n_depth:

        basemodel = xgb.XGBRegressor(n_estimators=Nest, max_depth=d, learning_rate=0.1)
        model_xgb = sk.MultiOutputRegressor(basemodel)
        #m = model_xgb.fit(stocks_lag5_train, stocks_lag5_label_train)
       
        tmpscore = np.mean(cross_val_score(model_xgb, stocks_lag, stocks_lag_label, cv=5, scoring = acc_scorer))
        
        if tmpscore > best_score:
            best_model = [Nest, d, tmpscore]
            best_score =  tmpscore
            
        score[str(Nest)+'_'+str(d)] = tmpscore



#for i in range(p_around_test)



    
    
    