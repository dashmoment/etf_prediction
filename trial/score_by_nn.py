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


srcPath = '../Data/0601/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath = '../Data/0601/all_meta_data_Nm_1_MinMax_120.pkl'
*_,meta = gu.read_metafile(metaPath) 
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)
single_stock = tv_gen._selectData2array(f,['0050'], None)

train_data = single_stock[:-210]
test_data = single_stock[-210:]

import random as rnd

train_sample_index = []
batch_size = 20000
period = 21

#Get train sample
samples = []
for _ in range(batch_size*2):
    train_sample_index = rnd.randint(0,len(train_data)-period)
    samples.append(train_data[train_sample_index:train_sample_index+21])
    
samples = np.stack(samples, axis=0)  
train_sample = np.reshape(samples[:,:16, 84:87], (2*batch_size, period-5, -1))
train_label = np.reshape(np.argmax(samples[:,-5:, -3:], axis=-1), (2*batch_size, 5, -1))   

train_sample = np.concatenate([train_sample[:batch_size], train_sample[batch_size:]], axis=2)  
train_label = np.concatenate([train_label[:batch_size], train_label[batch_size:]], axis=2)  

train_sample = np.reshape(train_sample, (batch_size, -1))
train_label = np.mean(np.equal(train_label[:,:,0], train_label[:,:,1]).astype(np.float64), axis=-1)

#Get test sample
test_sample = []
for idx in range(len(test_data)-21):
    test_sample.append(test_data[idx:idx+21])

match_samples = []
for _ in range(len(test_sample)):
    match_sample_index = rnd.randint(0,len(single_stock)-period)
    match_samples.append(single_stock[match_sample_index:match_sample_index+21])

test_sample = np.array(test_sample)
match_samples = np.array(match_samples)

test = np.stack([test_sample[:,:16,84:87], match_samples[:,:16,84:87]], axis=2)
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
model.add(layers.Dense(64, activation = "relu", input_shape=(96, )))
# Hidden - Layers
#model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(128, activation = "relu"))
#model.add(layers.Dropout(0.6))
model.add(layers.Dense(256, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

model.compile(
 optimizer = "adam",
 loss = losses.mean_squared_error,
 metrics = ["mse"]
)

results = model.fit(
 train_sample, train_label,
 epochs= 5000,
 batch_size = 128,
 validation_data = (test, test_label)
)






