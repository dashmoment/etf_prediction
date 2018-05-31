import sys
sys.path.append('../../')
import pandas as pd
import pickle

#**********Write to submit file********************
    

xgb_config = '../config/20180525/best_config_xgb_speicalDate.pkl'
rf_config = '../config/20180531/best_config_rf_speicalDate_npw_mfcont_cscore.pkl'
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


#xgb_dow = '/home/dashmoment/workspace/etf_prediction/trainer/config/20180526/best_config_xgb_speicalDate_nsw_npw_cscore.pkl'
xgb_nsnpw = '/home/ubuntu/shared/workspace/etf_prediction/trainer/config/20180526/best_config_xgb_speicalDate_nsw_npw_cscore.pkl'
f = open(xgb_nsnpw, 'rb')
xgb_nsnpw = pickle.load(f)

xgb_npw= '/home/ubuntu/shared/workspace/etf_prediction/trainer/config/20180526/best_config_xgb_speicalDate_npw_cscore.pkl'
f = open(xgb_npw, 'rb')
xgb_npw = pickle.load(f)