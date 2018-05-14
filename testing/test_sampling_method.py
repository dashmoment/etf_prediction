import sys
sys.path.append('../')
import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_specialList as dp
import model_zoo as mz
import loss_func as l

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


tv_gen = dp.train_validation_generaotr()
c = conf.config('test_sampleMethod').config['common']

process_data = tv_gen._load_data(c['src_file_path'])
select_data = tv_gen._selectData2array(process_data, c['input_stocks'], None)

#Test _selectData2array function is match with raw_data with shape (Period, features, c['input_stocks'])
for sIdx in range(len(c['input_stocks'])):
    raw_data = process_data.loc[c['input_stocks'][sIdx]]
    selected_data = select_data[:,:,sIdx]
    
    test_result = []
    for i in range(len(selected_data)):
    
        test_result.append((raw_data.iloc[i] == selected_data[i]).all())
        
    if np.array(test_result).any() == True:
        print('[Test] Stock {} feature Match test: Pass'.format(c['input_stocks'][sIdx]))
    else:
        print('[Test] Stock {} feature Match test: Fail'.format(c['input_stocks'][sIdx]))
        

#Test _split_train_val_side_by_side method
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])
total_len = len(select_data)
sample_window =  c['input_step']+ c['predict_step']
pivot = int(total_len*(1- c['train_eval_ratio']))

test_result = []

for i in range(pivot - sample_window):

    processed_train = train[i]
    raw_train = select_data[i:i+sample_window, :,:]
    
    test_result.append((processed_train == raw_train).all())
    
for i in range(pivot, len(select_data) - sample_window):

    processed_train = validation[i-pivot]
    raw_train = select_data[i:i+sample_window, :,:]
    
    test_result.append((processed_train == raw_train).all())
    
if np.array(test_result).all() == True:
        print('[Test] _split_train_val_side_by_side: Pass')
else:
        print('[Test] _split_train_val_side_by_side: Fail')
        
#Test single stock get cls batch
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], [c['input_stocks'][0]], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])
batch_train, batch_label = sesswrapper.get_batch_cls(train, c['input_step'], c['batch_size'], 0)

data_result = []
label_result = []

for i in range(c['batch_size']):
    train_data = select_data[i:i+c['input_step'],:,0]   
    b_data = batch_train[i]  
    train_label = select_data[i+c['input_step']:i+c['input_step']+c['predict_step'],-3:,0]  
    b_label = batch_label[i]  
    
    data_result.append((train_data==b_data).all())
    label_result.append((train_label==b_label).all())
    
if np.array(data_result).all() == True and np.array(label_result).all() == True:
        print('[Test] get_batch_cls: Pass')
else:
        print('[Test] get_batch_cls: Fail')


#======================================

import sys
sys.path.append('../')
import hparam as conf
import data_process_specialList as dp
c = conf.config('test_sampleMethod').config['common']
train_mStock, valid_mStock, train_s, valid_s, missing_feature = dp.read_special_data(c['input_step'], c['predict_step'],  c['train_eval_ratio'] ,c['src_file_path'])


#for s in train_s:
mStock_test = []
for sIdx in range(len(c['input_stocks'])):
    a = train_s[c['input_stocks'][sIdx]][2,:50, 0:5]  
    b = select_data[2:52, 0:5,sIdx]

    mStock_test.append((a == b).all())

if np.array(mStock_test).all() == True:
        print('[Test] mStock SpecialList: Pass')
else:
        print('[Test] mStock SpecialList: Fail')
