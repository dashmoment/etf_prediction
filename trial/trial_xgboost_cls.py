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


c = conf.config('trial_cnn_cls').config['common']
*_,meta = gu.read_metafile(c['meta_file_path'])
tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
data = tv_gen._selectData2array(f, f.index, None)

#Sample from stack last ud
train_sample = data[:-30]
flat_train_sample = np.reshape(np.transpose(train_sample, (0,2,1)), (-1,104))
test_sample = data[-30:]
flat_test_sample = np.reshape(np.transpose(test_sample, (0,2,1)), (-1,104))

#Sample from total data
#flat_data =  np.reshape(np.transpose(data, (0,2,1)), (-1,104))
#flat_train_sample = flat_data[:-800]
#flat_test_sample = flat_data[-800:]


#**********ratio***************
train = flat_train_sample[:,91:96] 
#train = prepro.normalize(train, axis = 0)
test  =  flat_test_sample[:,91:96] 
#test = prepro.normalize(test, axis = 0)
train_label = np.stack((flat_train_sample[:, -3] + flat_train_sample[:, -2] , flat_train_sample[:, -1]), axis=1)
test_label =  np.stack((flat_test_sample[:, -3] + flat_test_sample[:, -2], flat_test_sample[:, -1]) , axis=1)
#train_label = flat_train_sample[:, -3:]
#test_label = flat_test_sample[:,-3:]
train, train_label = data_label_shift(train, train_label, lag_day=1)
test, test_label = data_label_shift(test, test_label, lag_day=1)

predict_dict['ratio'] = example_xgb_v2(train, train_label, test, test_label, list(range(5)))

#*********velocity**********

restore_label = np.argmax(flat_test_sample[1:,-3:], axis=-1)
restore_predict = []
for i in range(len(predict[1])):
    if predict[1][i] == 1: restore_predict.append(2)
    else: restore_predict.append(0)
    
restore_predict = np.array(restore_predict)
print("Restore Accuracy: ", accuracy_score(restore_label, restore_predict))



