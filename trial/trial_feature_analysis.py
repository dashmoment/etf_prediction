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


c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']
*_,meta = gu.read_metafile(c['meta_file_path'])
tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
data = tv_gen._selectData2array(f, f.index, None)

#train = data[:,:-3,0]
#label = data[:,-3:,0]
#
##ratio
#train_ratio = train[1:]/train[:-1]
#label_ratio = label[1:]
#train_ratio, label_ratio = data_label_shift(train_ratio, label_ratio, lag_day=1)
#
#
#all_ratio = np.hstack((train_ratio, label_ratio))
#corr_ratio = get_corr_up_and_down(pd.DataFrame(all_ratio), [96, 98], [96,97,98])
#p =example_xgb(train_ratio, label_ratio, [68, 32, 93])
#
##minus
#train_minus = train[1:]-train[:-1]
#label_minus = label[1:]
#train_minus, label_minus = data_label_shift(train_minus, label_minus, lag_day=1)
#all_ratio = np.hstack((train_ratio, label_ratio))
#corr_minus = get_corr_up_and_down(pd.DataFrame(all_ratio), [96, 98], [96,97,98])
#p = example_xgb(train_minus, label_minus, [68])
#
##Add lag data
#lag = 60
#
#train_lag = []
#label_lag = label[lag:]
#for i in range(lag, len(train)):
#    train_lag.append(np.concatenate([train[i],train[i-lag]]))
#
#train_lag = np.vstack(train_lag)
#train_lag, label_lag = data_label_shift(train_lag, label_lag, lag_day=1)
#all_lag = np.hstack((train_lag, label_lag))
#corr_lag = get_corr_up_and_down(pd.DataFrame(all_lag), [192, 194], [192,193,194])
#p = example_xgb(train_lag, label_lag, [88,71,72,100,4])
#
#
#
##Only Up and down
#
#train_ud = label
#ud_feature = train_ud[1:] - train_ud[:-1]
#serial_feature = [] 
##[1,0,0]: two day down 
##[0,1,0]: two day fair
##[0,0,1]: two day up
#
#for i in range(1, len(train_ud)):
#    
#    if train_ud[i][0] == 1 and train_ud[i-1][0] == 1:
#        serial_feature.append([1,0,0])
#    
#    elif train_ud[i][1] == 1 and train_ud[i-1][1] == 1:
#        serial_feature.append([0,1,0])
#        
#    elif train_ud[i][2] == 1 and train_ud[i-1][2] == 1:
#        serial_feature.append([0,0,1])
#    else:
#        serial_feature.append([0,0,0])
#        
#        
#serial_feature_2 = [] 
##[1,0,0]: three day down 
##[0,1,0]: three day fair
##[0,0,1]: three day up
#
#for i in range(2, len(train_ud)):
#    
#    if train_ud[i][0] == 1 and train_ud[i-1][0] == 1 and train_ud[i-2][0] == 1:
#        serial_feature_2.append([1,0,0])
#    
#    elif train_ud[i][1] == 1 and train_ud[i-1][1] == 1 and train_ud[i-2][1] == 1:
#        serial_feature_2.append([0,1,0])
#        
#    elif train_ud[i][2] == 1 and train_ud[i-1][2] == 1 and train_ud[i-2][2] == 1:
#        serial_feature_2.append([0,0,1])
#    else:
#        serial_feature_2.append([0,0,0])
#        
#    
#
#serial_feature = np.vstack(serial_feature).astype(np.float64)
#serial_feature_2 = np.vstack(serial_feature_2).astype(np.float64)
#
#train_ud = np.hstack([train_ud[2:], ud_feature[1:],serial_feature[1:], serial_feature_2])
#label_ud = label[2:]
#
#train_ud, label_ud = data_label_shift(train_ud, label_ud, lag_day=1)
#all_ud = np.hstack((train_ud, label_ud))
#corr_ud = get_corr_up_and_down(pd.DataFrame(all_ud), [12, 14], [12,13,14])
#p = example_xgb(train_ud, label_ud, list(range(12)))
#
#plt.figure(figsize=(15,15))
#for i in range(1,15):
#    plt.subplot(4,4,i)
#    plt.scatter(train_ratio[:,i], np.argmax(label_ratio[:,-3:], axis=-1))
#
#
## Diff2
#data_raw = data[1:, 54:85]
#f_diff = data[1:, 54:85] - data[:-1, 54:85]
#f_diff2 = f_diff[1:, :] - f_diff[:-1, :]
#label_diff = label[2:]
#
#processed_data = np.hstack((data_raw[1:,:], f_diff[1:,:], f_diff2))
#train_diff, label_diff = data_label_shift(processed_data, label_diff, lag_day=1)
#p = example_xgb(train_diff, label_diff, [61, 59, 60])



#************curve expert*************
import seaborn as sns
import sklearn.preprocessing as prepro

#plt.plot(data[:,0,0])

flat_data = np.reshape(np.transpose(data, (0,2,1)), (-1,104))

#sns.distplot(flat_data[:,68])
#Baseline all feature - Validation Accuracy:  0.52625
base = flat_data
base = prepro.normalize(base, axis = 0)
base_label = flat_data[:, -3:]
base, base_label = data_label_shift(base, base_label, lag_day=1)
p_base = example_xgb(base, base_label, list(range(104)))

#kdj - Validation Accuracy:  0.59625
kdj = flat_data[:,66:70]
kdj = prepro.normalize(kdj, axis = 0)
kdj_label = flat_data[:, -3:]
kdj, kdj_label = data_label_shift(kdj, kdj_label, lag_day=1)
p_kdj = example_xgb(kdj, kdj_label, list(range(4)))

#MACD - Validation Accuracy:  0.50125
macd = flat_data[:,54:63]
macd = prepro.normalize(macd, axis = 0)
macd_label = flat_data[:, -3:]
macd, macd_label = data_label_shift(macd, macd_label, lag_day=1)

p_macd = example_xgb(macd, macd_label, list(range(9)))

#RSI - Validation Accuracy:  0.5225
rsi = flat_data[:,63:66]
rsi = prepro.normalize(rsi, axis = 0)
rsi_label = flat_data[:, -3:]
rsi, rsi_label = data_label_shift(rsi, rsi_label, lag_day=1)
p_rsi = example_xgb(rsi, rsi_label, list(range(3)))


#kdj + DOW - Validation Accuracy:  0.5875
data_dow = add_DOW(data)
flat_data_dow = np.reshape(np.transpose(data_dow, (0,2,1)), (-1,109))
kdj_dow = np.hstack((flat_data_dow[:, :5], flat_data[:,66:70]))
#kdj_dow = prepro.normalize(kdj_dow, axis = 0)
label_kdj_dow = flat_data[:, -3:]
kdj_dow, label_kdj_dow = data_label_shift(kdj_dow, label_kdj_dow, lag_day=1)
p_kdj_dow = example_xgb(kdj_dow, label_kdj_dow, list(range(9)))


#RSI + DOW - Validation Accuracy: 0.51625
rsi = np.hstack((flat_data_dow[:, :5], flat_data[:,63:66]))  
rsi = prepro.normalize(rsi, axis = 0)
rsi_label = flat_data[:, -3:]
rsi, rsi_label = data_label_shift(rsi, rsi_label, lag_day=1)
p_rsi_dow = example_xgb(rsi, rsi_label, list(range(7)))


#ratio - Validation Accuracy: 0.625
ratio = flat_data[:,91:96] 
ratio = prepro.normalize(ratio, axis = 0)
ratio_label = flat_data[:, -3:]
ratio, ratio_label = data_label_shift(ratio, ratio_label, lag_day=1)
p_ratio = example_xgb(ratio, ratio_label, list(range(5)))

#ratio+DOW - Validation Accuracy: 0.6225
data_ratiodow = add_DOW(data)
flat_data_ratiodow = np.reshape(np.transpose(data_ratiodow, (0,2,1)), (-1,109))
ratio_dow = np.hstack((flat_data_ratiodow[:, :5], flat_data[:,91:96]))
label_ratio_dow = flat_data[:, -3:]
p_ratiodow = feature_expert_trail(ratio_dow, list(range(10)), label_ratio_dow , normalize=True, axis=0)

#price - Validation Accuracy: 0.5275
p_price = feature_expert_trail(flat_data, list(range(0,5)), axis=1)


#taiex - Validation Accuracy: 0.5225
taiex = flat_data[:,87:91] 
taiex = prepro.normalize(taiex, axis = 0)
taiex_label = flat_data[:, -3:]
taiex, taiex_label = data_label_shift(taiex, taiex_label, lag_day=5)
p_taiexo = example_xgb(taiex, taiex_label, list(range(4)))

#MA - Validation Accuracy: 0.5225
ma = flat_data[:,5:50] 
ma = prepro.normalize(ma, axis = 0)
ma_label = flat_data[:, -3:]
ma, ma_label = data_label_shift(ma, ma_label, lag_day=5)
p_ma= example_xgb(ma, ma_label, list(range(45)))


#ATR Validation Accuracy:  0.45125
p_atr = feature_expert_trail(flat_data, list(range(75,78)), axis=0)
#NATR Validation Accuracy:  0.45125
p_natr = feature_expert_trail(flat_data, list(range(78,81)), axis=0)
#ATR+NATR
p_naatr = feature_expert_trail(flat_data, list(range(75,81)), axis=0)


#kdj + ratio Validation Accuracy:  0.64125
kgj_ratio_mask =  list(range(66, 70)) + list(range(91,96))
p_kdj_ratio = feature_expert_trail(flat_data, kgj_ratio_mask,lagday = 1)

#rsi + ratio validation Accuracy:  0.62375
rsi_ratio_mask = list(range(63,66)) + list(range(91,96)) 
p_rsi_ratio = feature_expert_trail(flat_data, rsi_ratio_mask)

#data_velocity Validation Accuracy:  0.6525
data_velocity= (data[1:,0:4] - data[:-1,0:4])/(data[:-1,0:4] + 0.1)
velocity_label = data[1:,-3:]
flat_data_v = np.reshape(np.transpose(data_velocity, (0,2,1)), (-1,4))
velocity_label = np.reshape(np.transpose(velocity_label, (0,2,1)), (-1,3))
p_velocity = feature_expert_trail(flat_data_v, list(range(5)), velocity_label, lagday=1)


#data_accerlation Validation Accuracy:  0.56375
data_velocity= (data[1:,0:4] - data[:-1,0:4])/(data[:-1,0:4] + 0.1)
data_acc = (data_velocity[1:,0:4] - data_velocity[:-1,0:4])/(data_velocity[:-1,0:4] + 0.1)
acc_label = data[2:,-3:]
flat_data_acc = np.reshape(np.transpose(data_acc, (0,2,1)), (-1,4))
label_acc = np.reshape(np.transpose(acc_label, (0,2,1)), (-1,3))
p_acc = feature_expert_trail(flat_data_acc, list(range(5)), label_acc, lagday=1)

#data_velocity + data_accerlation Validation Accuracy:  0.6525
data_velocity= (data[1:,0:4] - data[:-1,0:4])/(data[:-1,0:4] + 0.1)
data_acc = (data_velocity[1:,0:4] - data_velocity[:-1,0:4])/(data_velocity[:-1,0:4] + 0.1)
data_velocity = data_velocity[1:]
data_av = np.concatenate([data_acc, data_velocity], axis=1)
label_av = data[2:,-3:]
flat_data_av = np.reshape(np.transpose(data_av, (0,2,1)), (-1,8))
label_av = np.reshape(np.transpose(label_av, (0,2,1)), (-1,3))
p_av = feature_expert_trail(flat_data_av, list(range(8)), label_av, lagday=1, axis=0)


#ratio + velocity Validation Accuracy:  0.64625
data_velocity= (data[1:,0:4] - data[:-1,0:4])/(data[:-1,0:4] + 0.1)
data_ratio = data[1:,91:96] 
data_vr = np.concatenate([data_velocity, data_ratio], axis=1)
label_vr = data[1:,-3:]
flat_data_vr = np.reshape(np.transpose(data_vr, (0,2,1)), (-1,9))
label_vr = np.reshape(np.transpose(label_vr, (0,2,1)), (-1,3))
p_vr = feature_expert_trail(flat_data_vr, list(range(9)), label_vr, lagday=1, axis=0)


#kdj + velocity Validation Accuracy:  0.65625
data_velocity= (data[1:,0:4] - data[:-1,0:4])/(data[:-1,0:4] + 0.1)
data_kdj = data[1:,66:70] 
data_vkdj = np.concatenate([data_velocity, data_kdj], axis=1)
label_vkdj = data[1:,-3:]
flat_data_vkdj = np.reshape(np.transpose(data_vkdj, (0,2,1)), (-1,8))
label_vkdj = np.reshape(np.transpose(label_vkdj, (0,2,1)), (-1,3))
p_vkdj = feature_expert_trail(flat_data_vkdj, list(range(9)), label_vkdj, lagday=1, axis=0)


#ratio_velocity Validation Accuracy:  0.5775
data_r_velocity= (data[1:,91:96] - data[:-1,91:96])/(data[:-1,91:96] + 0.1)
rvelocity_label = data[1:,-3:]
flat_data_rv = np.reshape(np.transpose(data_r_velocity, (0,2,1)), (-1,5))
rvelocity_label = np.reshape(np.transpose(rvelocity_label, (0,2,1)), (-1,3))
p_r_velocity = feature_expert_trail(flat_data_rv, list(range(5)), rvelocity_label, lagday=1)

#kdj + MACD + rssi + ratio Validation Accuracy:  0.64125s
kmrr_mask =  list(range(66, 70)) + list(range(63,66)) + list(range(91,96)) +  list(range(54,62))
p_rkmrr = feature_expert_trail(flat_data, kmrr_mask, lagday=1)

#ud Validation Accuracy:   0.6425s
ud_mask =  list(range(101, 104))
p_ud = feature_expert_trail(flat_data, ud_mask, lagday=1)


