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
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94_0050.pkl')
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94_0050.pkl')
#data_path = '/home/ubuntu/dataset/etf_prediction/raw_data/tetfp.csv'

#*_,meta = gu.read_metafile('/home/dashmoment/workspace/etf_prediction/Data/all_meta_data_Nm_1_MinMax_94.pkl')
#f = tv_gen._load_data('/home/dashmoment/workspace/etf_prediction/Data/all_feature_data_Nm_1_MinMax_94.pkl')
data_path = '/home/dashmoment/workspace/etf_prediction/Data/raw_data/20180518/tetfp.csv'

def change_raw_columns_name(inputfile:pd.DataFrame):
    inputfile.columns = ["ID", "Date", "name", "open_price", "max", "min", "close_price", "trade"]
    

tasharep = pd.read_csv(data_path, encoding = "big5-hkscs", dtype=str).dropna(axis = 1) # training data
change_raw_columns_name(tasharep)


tasharep.ID = tasharep.ID.astype(str)
tasharep.Date = tasharep.Date.astype(str)

tasharep["ID"] = tasharep["ID"].astype(str)
tasharep["ID"] = tasharep["ID"].str.strip()

tasharep["Date"] = tasharep["Date"].astype(str)
tasharep["Date"] = tasharep["Date"].str.strip()

tasharep["open_price"] = tasharep["open_price"].astype(str)
tasharep["open_price"] = tasharep["open_price"].str.strip()
tasharep["open_price"] = tasharep["open_price"].str.replace(",", "")
tasharep["open_price"] = tasharep["open_price"].astype(float)

tasharep["max"] = tasharep["max"].astype(str)
tasharep["max"] = tasharep["max"].str.strip()
tasharep["max"] = tasharep["max"].str.replace(",", "")
tasharep["max"] = tasharep["max"].astype(float)

tasharep["min"] = tasharep["min"].astype(str)
tasharep["min"] = tasharep["min"].str.strip()
tasharep["min"] = tasharep["min"].str.replace(",", "")
tasharep["min"] = tasharep["min"].astype(float)

tasharep["close_price"] = tasharep["close_price"].astype(str)
tasharep["close_price"] = tasharep["close_price"].str.strip()
tasharep["close_price"] = tasharep["close_price"].str.replace(",", "")
tasharep["close_price"] = tasharep["close_price"].astype(float)

tasharep["trade"] = tasharep["trade"].astype(str)
tasharep["trade"] = tasharep["trade"].str.strip()
tasharep["trade"] = tasharep["trade"].str.replace(",", "")
tasharep["trade"] = tasharep["trade"].astype(float)

# Get the ID list
tasharep_ID = tasharep.ID.unique()

# Get the Date list
Date = tasharep.Date.unique()
tasharep_ID_group = tasharep.groupby("ID")

price_list = {}
for i in range(len(tasharep_ID)):
    
    idx = tasharep_ID_group.groups[tasharep_ID[i]]
    s = tasharep.iloc[idx]
    stock_price = s.close_price[-5:]
   
    price_list[tasharep_ID[i]] = [np.mean(stock_price) for i in range(5)]
    


#****************Get ud*****************
    
#def classification_train():
import trial_xgboost_ensCls as ens


clean_stocks = {}
MtestSamples = 30
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

    single_stock = tv_gen._selectData2array(f, [s],  ['20150207','20160118'])
    
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

   # return [test_label, test_label_restore ]
    
    
p = classification_train()
    

def get_model(lag_day):

    
    
    clean_stocks = {}
    MtestSamples = 0
    
 
    #kgj + ratio + ud 
    mask_list = list(range(66, 70)) + list(range(81,86))  + list(range(91, 94))
    mask = []
    for i in range(94):
        if i in mask_list: mask.append(True)
        else: mask.append(False)
    
    for s in f.index:
    
        single_stock = tv_gen._selectData2array(f, [s], None)
        
        tmpStock = []
        for i in range(len(single_stock)):
            if not np.isnan(single_stock[i,0:5]).all():
                tmpStock.append(single_stock[i])
        single_stock = np.array(tmpStock)
        data_velocity= (single_stock[1:,0:4] - single_stock[:-1,0:4])/(single_stock[:-1,0:4] + 0.1)
        data = single_stock[1:]
        
        delay_data = data[:-lag_day]
        delay_data_velocity = data_velocity[:-lag_day]
        label = data[lag_day:, -3:]
        
        clean_stocks[s] = { 'train': delay_data,
                            'train_velocity': delay_data_velocity,
                            'train_label': label}
                           
    
    train = {}
    train['train'] = np.concatenate([clean_stocks[s]['train'] for s in  clean_stocks], axis=0)
    train['train_velocity'] = np.concatenate([clean_stocks[s]['train_velocity'] for s in  clean_stocks], axis=0)
    train['train_label'] = np.concatenate([clean_stocks[s]['train_label'] for s in  clean_stocks], axis=0)
    
    
    train_fe = ens.feature_extractor(train['train'], train['train_velocity'] )
    train_label_raw = np.stack((train['train_label'][:, -3] + train['train_label'][:, -2] , train['train_label'][:, -1]), axis=1)
    
    #train_label_raw = train['train_label']
    #test_label_raw = test['test_label']
    
    train_data = train_fe.ratio_velocity()
    train_label = np.argmax(train_label_raw, axis=-1)
    
    
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False)
    m = model.fit(train_data, train_label)
    
    
    return m


import trial_xgboost_ensCls as ens
m1 = get_model(1)
m2 = get_model(2)
m3 = get_model(3)
m4 = get_model(4)
m5 = get_model(5)


clean_stocks = {}
MtestSamples = 0

 
#kgj + ratio + ud 
mask_list = list(range(66, 70)) + list(range(81,86))  + list(range(91, 94))
mask = []
for i in range(94):
    if i in mask_list: mask.append(True)
    else: mask.append(False)

predict_cls = {}

for s in f.index:

    single_stock = tv_gen._selectData2array(f, [s], None)
    
    tmpStock = []
    for i in range(len(single_stock)):
        if not np.isnan(single_stock[i,0:5]).all():
            tmpStock.append(single_stock[i])
    single_stock = np.array(tmpStock)
    data_velocity= (single_stock[1:,0:4] - single_stock[:-1,0:4])/(single_stock[:-1,0:4] + 0.1)
    data = single_stock[1:]
    
    delay_data = data
    delay_data_velocity = data_velocity
    label = data[:, -3:]
    
    train_fe = ens.feature_extractor(delay_data, delay_data_velocity )
    train_data = train_fe.ratio_velocity()
    
    inputs = np.reshape(train_data[-1], (1,-1))
    p1 = m1.predict(inputs)
    p2 = m1.predict(inputs)
    p3 = m1.predict(inputs)
    p4 = m1.predict(inputs)
    p5 = m1.predict(inputs)
    
    
    tmp_p = np.array([p1, p2, p3, p4, p5])
 
    restore = []
    
    for i in range(5):
        if tmp_p[i] == 0:
            restore.append(-1)
        else:
            restore.append(1)
            
    predict_cls[s] = np.array(restore)


#**********Write to submit file********************
    
columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
          'Wed_ud', 'Wed_cprice	', 'Thu_ud', 'Thu_cprice', 'Fri_ud',	'Fri_cprice']

df = pd.DataFrame(columns=columns)  
idx = 0

for s in f.index:     
    results = [s]
    for i in range(5):
        results.append(predict_cls[s][i])
        results.append(price_list[s][i])
        
    
    df.loc[idx] = results
    idx+=1

df = df.set_index('ETFid') 
df.to_csv('../submission/submit_20180520.csv', sep=',')


#raw = tv_gen._selectData2array(f, f.index, None)
#
#fe_train = ens.feature_extractor(data, data_velocity)
#d_ratio = fe_train.ud()
##p1 = m_1['ratio'].predict(np.reshape(d_ratio[-1], (1,-1)))
#input_data = np.reshape(d_ratio[-1], (1,-1))
#p1 = m_1['ud'].predict(input_data)
#p2 = m_2['ud'].predict(input_data)
#p3 = m_3['ratio'].predict(input_data)
#p4 = m_4['ratio'].predict(input_data)
#p5 = m_5['ratio'].predict(input_data)
#


#*************Test model is valid***************
#Lagday = 2
#
#data = tv_gen._selectData2array(f, f.index, None)
#noraml_data = data[:,:,:14] 
#special_data = data[:,:,14:] 
#
#data_velocity_= (noraml_data[1:,0:4] - noraml_data[:-1,0:4])/(noraml_data[:-1,0:4] + 0.1)
#noraml_data = noraml_data[1:]
#
#train_sample = noraml_data[:-30]
#train_sample_v = data_velocity[:-30]
#flat_train_sample = np.reshape(np.transpose(noraml_data, (0,2,1)), (-1,104))
#flat_train_sample_velocity =  np.reshape(np.transpose(data_velocity, (0,2,1)), (-1,4))
#test_sample = data[-30:]
#test_sample_v = data_velocity[-30:]
#flat_test_sample = np.reshape(np.transpose(test_sample, (0,2,1)), (-1,104))
#flat_test_sample_velocity = np.reshape(np.transpose(test_sample_v, (0,2,1)), (-1,4))
#
#
#fe_train = ens.feature_extractor(flat_train_sample, flat_train_sample_velocity)
#d_ratio = fe_train.ratio()
#fe_test = ens.feature_extractor(flat_test_sample, flat_test_sample_velocity)
#d_ratio_test = fe_test.ratio()
#
#train_label_raw = np.stack((flat_train_sample[:, -3] + flat_train_sample[:, -2] , flat_train_sample[:, -1]), axis=1)
#test_label_raw =  np.stack((flat_test_sample[:, -3] + flat_test_sample[:, -2], flat_test_sample[:, -1]) , axis=1)
#
#
#train, train_label = data_label_shift(d_ratio, train_label_raw, lag_day=Lagday)
#test, test_label = data_label_shift(d_ratio_test, test_label_raw, lag_day=Lagday)
#train_label = np.argmax(train_label, axis=-1)
#test_label = np.argmax(test_label, axis=-1)
#
#y_xgb_train = m_1['ratio'].predict(train)
#y_xgb_v = m_1['ratio'].predict(test)
#
#
#print("Train Accuracy [ratio]: ", accuracy_score(y_xgb_train, train_label))
#print("Validation Accuracy [ratio]: ",accuracy_score(y_xgb_v, test_label))


