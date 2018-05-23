import sys
sys.path.append('../../')
import pandas as pd
import pickle

#**********Write to submit file********************
    

dow_config = '../config/best_config_dow.pkl'
normal_config = '../config/best_config_normal.pkl'
special_config = '../config/best_config_speicalDate.pkl'

f = open(dow_config, 'rb')
dow_config = pickle.load(f)

f = open(normal_config, 'rb')
normal_config = pickle.load(f)


f = open(special_config, 'rb')
special_config = pickle.load(f)