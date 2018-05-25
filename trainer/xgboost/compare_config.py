import sys
sys.path.append('../../')
import pandas as pd
import pickle

#**********Write to submit file********************
    

xgb_config = '../config/20180525/best_config_xgb_speicalDate.pkl'
rf_config = '../config/20180525/best_config_rf_speicalDate.pkl'
svc_config = '../config/20180525/best_config_svc_speicalDate.pkl'
stack_config = '../config/20180525/best_config_stack_speicalDate.pkl'


f = open(xgb_config, 'rb')
xgb_config = pickle.load(f)

f = open(rf_config, 'rb')
rf_config = pickle.load(f)


f = open(svc_config, 'rb')
svc_config = pickle.load(f)

f = open(stack_config, 'rb')
stack_config = pickle.load(f)