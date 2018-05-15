import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l

tf.reset_default_graph()  
c = conf.config('trial_cnn_cls').config['common']
sample_window = c['input_step'] + c['predict_step']

tv_gen = dp.train_validation_generaotr()

f = tv_gen._load_data(c['src_file_path'])
stock = tv_gen._selectData2array(f, ['2330'], None)

train = stock[:769]
validation = stock[769:]

train_data = np.reshape(train[:,:96], (-1,96,1))
train_label = train[:,96:]
validation_data = np.reshape(validation[:,:96], (-1,96,1))
validation_label = validation[:,96:]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

mask = np.zeros((99), dtype=bool)
mask[:96] = True
mask[96:] = True
price = train[:, mask]
price_df = pd.DataFrame(price)

corr = price_df.corr()
sns.heatmap(corr)

plt.plot(preprocessing.scale(price[:,3]))
plt.scatter(list(range(len(price))),np.argmax(price[:,-3:], axis=-1))

label = train[:,-3:]
train_ = np.zeros(np.shape(train))

for i in range(1, len(train)):
    train_[i] = train[i] - train[i-1]

train_ = train_[1:len(train)-1, mask]
label_ = label[2:]
price_df_ = pd.DataFrame(train_)
corr = price_df_.corr()
sns.heatmap(corr)

plt.plot(preprocessing.scale(train_[0:10,3]))
plt.scatter(list(range(10)),np.argmax(label_[0:10,-3:], axis=-1))