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

#Train ratio
train_ratio = train[1:]/train[:-1]
label_ratio = label[1:]
train_ratio, label_ratio = data_label_shift(train_ratio, label_ratio, lag_day=1)


all_ratio = np.hstack((train_ratio, label_ratio))
corr_ratio = get_corr_up_and_down(pd.DataFrame(all_ratio), [86, 88], [86,87,88])
p =example_xgb(train_ratio, label_ratio, [84,82,69])

#Auto correlation
lag = 30
s1 = data[lag:]
s2 = data[:-lag]

ms1 = np.mean(s1)
ms2 = np.mean(s2)
ds1 = s1 - ms1
ds2 = s2 - ms2
divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
auto_corr = np.sum(ds1 * ds2) / divider















