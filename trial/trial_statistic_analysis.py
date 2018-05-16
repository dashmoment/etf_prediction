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

tv_gen = dp.train_validation_generaotr()
meta = gu.read_metafile(c['meta_file_path'])
f = tv_gen._load_data(c['src_file_path'])
data = tv_gen._selectData2array(f, ['0050'], None)

total_down = np.mean(data[:,-3])
total_fair = np.mean(data[:,-2])
total_up = np.mean(data[:,-1])

uu = 0
ud = 0
du = 0
dd = 0
fu = 0
fd = 0
ff = 0 

data_shift = data[1:,-3:] + data[:-1,-3:]

for i in range(len(data_shift)):
    if data_shift[i,-3] == 2: dd+=1
    elif data_shift[i,-2] == 2: ff+=1
    elif data_shift[i,-1] == 2: uu+=1


