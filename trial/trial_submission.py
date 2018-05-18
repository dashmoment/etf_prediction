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
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_104s.pkl')
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_104s.pkl')

def change_raw_columns_name(inputfile:pd.DataFrame):
    inputfile.columns = ["ID", "Date", "name", "open_price", "max", "min", "close_price", "trade"]
    
data_path = '/home/ubuntu/dataset/etf_prediction/raw_data/tetfp.csv'
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
import trial_xgboost_ensCls as ens


m_1 = ens.get_ens_model(1)
m_2 = ens.get_ens_model(2)
m_3 = ens.get_ens_model(3)
m_4 = ens.get_ens_model(4)
m_5 = ens.get_ens_model(5)

single_stock = tv_gen._selectData2array(f, ['006203'], None)
data_velocity= (single_stock[1:,0:4] - single_stock[:-1,0:4])/(single_stock[:-1,0:4] + 0.1)
data = single_stock[1:]

fe_train = ens.feature_extractor(data, data_velocity)
d_ratio = fe_train.ratio()
#p1 = m_1['ratio'].predict(np.reshape(d_ratio[-1], (1,-1)))
p1 = m_1['ratio'].predict(d_ratio)
p2 = m_2['ratio'].predict(d_ratio)
p3 = m_3['ratio'].predict(d_ratio)
p4 = m_4['ratio'].predict(np.reshape(d_ratio[-1], (1,-1)))
p5 = m_5['ratio'].predict(np.reshape(d_ratio[-1], (1,-1)))



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


