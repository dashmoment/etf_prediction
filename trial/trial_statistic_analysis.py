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
from sklearn import preprocessing

from utility_trial import *



tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94.pkl')
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94.pkl')

stock_list =  ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', 
               '006203', '006204', '006208','00690', '00692', '00701', '00713']


period = ['20130101', '20180520']


prob_ud = {}
for s in stock_list:
    data = tv_gen._selectData2array(f, [s], period)
    prob_ud[s] = [np.mean(data[:,-3]), np.mean(data[:,-2]), np.mean(data[:,-1])]


df = pd.DataFrame({'date':f.columns})
df['date'] = pd.to_datetime(df['date'])
df['dow'] = df['date'].dt.dayofweek


#*********Prob of ud by dow****************

stocks_ud = {}

for s in f.index:

    single_stock = tv_gen._selectData2array(f, [s], period)
    
    tmpStock = []
    for i in range(len(single_stock)):
        if not np.isnan(single_stock[i,0:5]).all():
            tmpStock.append(single_stock[i])
    single_stock = np.array(tmpStock)


    #single_stock = tv_gen._selectData2array(f, ['0050'], None)
    
    dow_array = np.reshape(np.array(df['dow'][-len(single_stock):]), (-1,1))
   
    
    label_ud = np.argmax(single_stock[:,-3:], axis=-1)
    
    data_ud = {'mon':[],
               'tue':[],
               'wed':[],
               'thu':[],
               'fri':[]       
                }
    
    
    for i in range(len(label_ud)):
        
        if dow_array[i][0] == 0: data_ud['mon'].append(label_ud[i])
        elif dow_array[i][0] == 1: data_ud['tue'].append(label_ud[i])
        elif dow_array[i][0] == 2: data_ud['wed'].append(label_ud[i])
        elif dow_array[i][0] == 3: data_ud['thu'].append(label_ud[i])
        elif dow_array[i][0] == 4: data_ud['fri'].append(label_ud[i])
    
    
    dist_ud = {'mon':{},
               'tue':{},
               'wed':{},
               'thu':{},
               'fri':{}       
                }
    
    for k in data_ud:
        dist_ud[k]['down'] = np.mean(np.equal(data_ud[k], 0))
        dist_ud[k]['fair'] = np.mean(np.equal(data_ud[k], 1))
        dist_ud[k]['up'] = np.mean(np.equal(data_ud[k], 2))
        
    stocks_ud[s] = dist_ud


