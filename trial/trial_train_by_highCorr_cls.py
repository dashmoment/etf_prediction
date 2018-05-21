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
import trial_xgboost_ensCls as ens

tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94_0050.pkl')
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94_0050.pkl')
#data_path = '/home/ubuntu/dataset/etf_prediction/raw_data/tetfp.csv'




clean_stocks = {}
MtestSamples = 20
lag_day = 1


#kgj + ratio + ud 
mask_list = list(range(66, 70)) + list(range(81,86))  + list(range(91, 94))
mask = []
for i in range(94):
    if i in mask_list: mask.append(True)
    else: mask.append(False)

stocks = ['0050']
#stocks = f.index

for s in stocks:

    single_stock = tv_gen._selectData2array(f, [s],  ['20150207','20160321'])
    
    tmpStock = []
    for i in range(len(single_stock)):
        if not np.isnan(single_stock[i,0:5]).all():
            tmpStock.append(single_stock[i])
    single_stock = np.array(tmpStock)
    data_velocity= (single_stock[1:,0:4] - single_stock[:-1,0:4])/(single_stock[:-1,0:4] + 0.1)
    data = single_stock[1:]
    
#    delay_data = prepro.normalize(data[:-lag_day], axis=0)
#    delay_data_velocity = prepro.normalize(data_velocity[:-lag_day], axis=0)
    delay_data = data[:-lag_day]
    delay_data_velocity = data_velocity[:-lag_day]
    label = data[lag_day:, -3:]
    
    clean_stocks[s] = { 'train': delay_data[:-MtestSamples],
                        'train_velocity': delay_data_velocity[:-MtestSamples],
                        'train_label': label[:-MtestSamples],
                        'test':delay_data[-MtestSamples:],
                        'test_velocity': delay_data_velocity[-MtestSamples:],
                        'test_label': label[-MtestSamples:]}

train = {}
test = {}
train['train'] = np.concatenate([clean_stocks[s]['train'] for s in  clean_stocks], axis=0)
train['train_velocity'] = np.concatenate([clean_stocks[s]['train_velocity'] for s in  clean_stocks], axis=0)
train['train_label'] = np.concatenate([clean_stocks[s]['train_label'] for s in  clean_stocks], axis=0)
test['test'] = np.concatenate([clean_stocks[s]['test'] for s in  ['0050']], axis=0)
test['test_velocity'] = np.concatenate([clean_stocks[s]['test_velocity'] for s in  ['0050']], axis=0)
test['test_label'] = np.concatenate([clean_stocks[s]['test_label'] for s in  ['0050']], axis=0)


train_fe = ens.feature_extractor(train['train'], train['train_velocity'] )
test_fe = ens.feature_extractor(test['test'], test['test_velocity'] )

train_label_raw = np.stack((train['train_label'][:, -3] + train['train_label'][:, -2] , train['train_label'][:, -1]), axis=1)
test_label_raw =  np.stack((test['test_label'][:, -3] + test['test_label'][:, -2], test['test_label'][:, -1]) , axis=1)

#train_label_raw = train['train_label']
#test_label_raw = test['test_label']

train_data = train_fe.ratio_velocity()
train_label = np.argmax(train_label_raw, axis=-1)
test_data = test_fe.ratio_velocity()
test_label = np.argmax(test_label_raw, axis=-1)


model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False)
model.fit(train_data, train_label)
y_xgb_train = model.predict(train_data)
y_xgb_valid = model.predict(test_data)
        
print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label))
print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_valid, test_label))

test_label = np.argmax(test['test_label'], axis=-1)
test_label_restore = []

for i in range(len(y_xgb_valid)):
    
    if y_xgb_valid[i] == 0: 
        test_label_restore.append(0)
    else:
        test_label_restore.append(2)

print("Test Accuracy [ratio]: ",accuracy_score(test_label_restore, test_label))


#***************Get recent data for test******************
clean_stocks_test = {}
for s in stocks:

    single_stock = tv_gen._selectData2array(f, [s],  ['20180321','20180504'])
    
    tmpStock = []
    for i in range(len(single_stock)):
        if not np.isnan(single_stock[i,0:5]).all():
            tmpStock.append(single_stock[i])
    single_stock = np.array(tmpStock)
    data_velocity= (single_stock[1:,0:4] - single_stock[:-1,0:4])/(single_stock[:-1,0:4] + 0.1)
    data = single_stock[1:]
    
#    delay_data = prepro.normalize(data[:-lag_day], axis=0)
#    delay_data_velocity = prepro.normalize(data_velocity[:-lag_day], axis=0)
    delay_data = data[:-lag_day]
    delay_data_velocity = data_velocity[:-lag_day]
    label = data[lag_day:, -3:]
    
    clean_stocks_test[s] = { 'train': delay_data,
                            'train_velocity': delay_data_velocity,
                            'train_label': label}
                      
train = {}
train['train'] = np.concatenate([clean_stocks_test[s]['train'] for s in  clean_stocks_test], axis=0)
train['train_velocity'] = np.concatenate([clean_stocks_test[s]['train_velocity'] for s in  clean_stocks_test], axis=0)
train['train_label'] = np.concatenate([clean_stocks_test[s]['train_label'] for s in  clean_stocks_test], axis=0)
train_fe = ens.feature_extractor(train['train'], train['train_velocity'] )
train_label_raw = np.stack((train['train_label'][:, -3] + train['train_label'][:, -2] , train['train_label'][:, -1]), axis=1)
train_data = train_fe.ratio_velocity()
train_label = np.argmax(train_label_raw, axis=-1)

y_xgb_train = model.predict(train_data)
        
print("Test 2018 Accuracy [ratio]: ",accuracy_score(y_xgb_train, train_label))













   # return [test_label, test_label_restore ]