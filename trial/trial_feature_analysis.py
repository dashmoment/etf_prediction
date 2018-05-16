import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from utility_trial import *


c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
data = tv_gen._selectData2array(f, ['0050'], None)

train = data[:,:-3]
label = data[:,-3:]

#ratio
train_ratio = train[1:]/train[:-1]
label_ratio = label[1:]
train_ratio, label_ratio = data_label_shift(train_ratio, label_ratio, lag_day=1)


all_ratio = np.hstack((train_ratio, label_ratio))
corr_ratio = get_corr_up_and_down(pd.DataFrame(all_ratio), [96, 98], [96,97,98])
p =example_xgb(train_ratio, label_ratio, [68, 32, 93])

#minus
train_minus = train[1:]-train[:-1]
label_minus = label[1:]
train_minus, label_minus = data_label_shift(train_minus, label_minus, lag_day=1)
all_ratio = np.hstack((train_ratio, label_ratio))
corr_minus = get_corr_up_and_down(pd.DataFrame(all_ratio), [96, 98], [96,97,98])
p = example_xgb(train_minus, label_minus, [68])

#Add lag data
lag = 60

train_lag = []
label_lag = label[lag:]
for i in range(lag, len(train)):
    train_lag.append(np.concatenate([train[i],train[i-lag]]))

train_lag = np.vstack(train_lag)
train_lag, label_lag = data_label_shift(train_lag, label_lag, lag_day=1)
all_lag = np.hstack((train_lag, label_lag))
corr_lag = get_corr_up_and_down(pd.DataFrame(all_lag), [192, 194], [192,193,194])
p = example_xgb(train_lag, label_lag, [88,71,72,100,4])



#Only Up and down

train_ud = label
ud_feature = train_ud[1:] - train_ud[:-1]
serial_feature = [] 
#[1,0,0]: two day down 
#[0,1,0]: two day fair
#[0,0,1]: two day up

for i in range(1, len(train_ud)):
    
    if train_ud[i][0] == 1 and train_ud[i-1][0] == 1:
        serial_feature.append([1,0,0])
    
    elif train_ud[i][1] == 1 and train_ud[i-1][1] == 1:
        serial_feature.append([0,1,0])
        
    elif train_ud[i][2] == 1 and train_ud[i-1][2] == 1:
        serial_feature.append([0,0,1])
    else:
        serial_feature.append([0,0,0])
        
        
serial_feature_2 = [] 
#[1,0,0]: three day down 
#[0,1,0]: three day fair
#[0,0,1]: three day up

for i in range(2, len(train_ud)):
    
    if train_ud[i][0] == 1 and train_ud[i-1][0] == 1 and train_ud[i-2][0] == 1:
        serial_feature_2.append([1,0,0])
    
    elif train_ud[i][1] == 1 and train_ud[i-1][1] == 1 and train_ud[i-2][1] == 1:
        serial_feature_2.append([0,1,0])
        
    elif train_ud[i][2] == 1 and train_ud[i-1][2] == 1 and train_ud[i-2][2] == 1:
        serial_feature_2.append([0,0,1])
    else:
        serial_feature_2.append([0,0,0])
        
    

serial_feature = np.vstack(serial_feature).astype(np.float64)
serial_feature_2 = np.vstack(serial_feature_2).astype(np.float64)

train_ud = np.hstack([train_ud[2:], ud_feature[1:],serial_feature[1:], serial_feature_2])
label_ud = label[2:]

train_ud, label_ud = data_label_shift(train_ud, label_ud, lag_day=1)
all_ud = np.hstack((train_ud, label_ud))
corr_ud = get_corr_up_and_down(pd.DataFrame(all_ud), [12, 14], [12,13,14])
p = example_xgb(train_ud, label_ud, list(range(12)))

plt.figure(figsize=(15,15))
for i in range(1,15):
    plt.subplot(4,4,i)
    plt.scatter(train_ud[:,i], np.argmax(label_ud[:,-3:], axis=-1))


# Diff2
data_raw = data[1:, 54:85]
f_diff = data[1:, 54:85] - data[:-1, 54:85]
f_diff2 = f_diff[1:, :] - f_diff[:-1, :]
label_diff = label[2:]

processed_data = np.hstack((data_raw[1:,:], f_diff[1:,:], f_diff2))
train_diff, label_diff = data_label_shift(processed_data, label_diff, lag_day=1)
all_diff = np.hstack((train_diff, label_diff))
corr_diff = get_corr_up_and_down(pd.DataFrame(all_diff), [93, 95], [93,94,95])

p = example_xgb(train_diff, label_diff, [61, 59, 60])



