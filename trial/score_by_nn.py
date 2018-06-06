import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp

srcPath = '/home/ubuntu/dataset/etf_prediction/0601/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath =  '/home/ubuntu/dataset/etf_prediction/0601/all_meta_data_Nm_1_MinMax_120.pkl'
#srcPath = '../Data/0601/all_feature_data_Nm_1_MinMax_120.pkl'
#metaPath = '../Data/0601/all_meta_data_Nm_1_MinMax_120.pkl'
*_,meta = gu.read_metafile(metaPath) 
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

single_stock = tv_gen._selectData2array(f,['0050'], None)
test_data = single_stock[-210:]

import random as rnd

batch_size = 1000
period = 10

#Get train sample

ms_sample = []
ms_label = []

for s in f.index:

    single_stock = tv_gen._selectData2array(f,[s], None)
    single_stock = dp.clean_stock(single_stock,meta, ['ma'])  
    if s == '0050': train_data = single_stock[:-210]
    else: train_data = single_stock
    
    samples = []
    for _ in range(batch_size*2):
        train_sample_index = rnd.randint(0,len(train_data)-period)
        samples.append(train_data[train_sample_index:train_sample_index+period])
        
    samples = np.stack(samples, axis=0)  
    train_sample = np.reshape(samples[:,:period-5, 5:49], (2*batch_size, period-5, -1))
    train_label = np.reshape(np.argmax(samples[:,-5:, -3:], axis=-1), (2*batch_size, 5, -1))   
    
    train_sample = np.concatenate([train_sample[:batch_size], train_sample[batch_size:]], axis=2)  
    train_label = np.concatenate([train_label[:batch_size], train_label[batch_size:]], axis=2)  
    
    train_sample = np.reshape(train_sample, (batch_size, -1))
    train_label = np.mean(np.equal(train_label[:,:,0], train_label[:,:,1]).astype(np.float64), axis=-1)
    
    ms_sample.append(train_sample)
    ms_label.append(np.reshape(train_label, (-1, 1)))
    
ms_sample = np.vstack(ms_sample)
ms_label =  np.reshape(np.vstack(ms_label), (-1))

#Get test sample
test_sample = []
for idx in range(len(test_data)-period):
    test_sample.append(test_data[idx:idx+period])

match_samples = []
for _ in range(len(test_sample)):
    match_sample_index = rnd.randint(0,len(single_stock)-period)
    match_samples.append(single_stock[match_sample_index:match_sample_index+period])

test_sample = np.array(test_sample)
match_samples = np.array(match_samples)

test = np.stack([test_sample[:,:period-5,5:49], match_samples[:,:period-5,5:49]], axis=2)
test_label = np.stack([np.argmax(test_sample[:,-5:,-3:], axis=-1), 
                       np.argmax(match_samples[:,-5:,-3:], axis=-1)], 
                        axis=2)
test = np.reshape(test, (len(test), -1))
test_label = np.mean(np.equal(test_label[:,:,0], test_label[:,:,1]).astype(np.float64), axis=-1)

#Build model
from keras import models
from keras import layers
from keras import losses

model = models.Sequential()
model.add(layers.Dense(128, activation = "relu", input_shape=(ms_sample.shape[-1], )))
# Hidden - Layers
#model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(128, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

model.compile(
 optimizer = "adam",
 loss = losses.mean_squared_error,
 metrics = ["mse"]
)

results = model.fit(
 ms_sample, ms_label,
 epochs= 5000,
 batch_size = 128,
 validation_data = (test, test_label)
)

#model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                                     learning_rate=0.05, max_depth=3, 
#                                     min_child_weight=1.7817, n_estimators=1000,
#                                     reg_alpha=0.4640, reg_lambda=0.8571,
#                                     subsample=0.5213, silent=1,
#                                     random_state =7, nthread = -1)
#
#
#model_xgb.fit(ms_sample, ms_label)
#p_train = model_xgb.predict(ms_sample)
#p_test = model_xgb.predict(test)
#mse_train = ((p_train - ms_label)**2).mean()
#mse_test = ((p_test - test_label)**2).mean()


