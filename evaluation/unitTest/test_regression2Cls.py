import sys
sys.append('../')
sys.append('../../')
import evaluation_2model as eval
import hparam as conf
import data_process_list as dp

tv_gen = dp.train_validation_generaotr()
conf_reg = conf.config('test_onlyEnc_biderect_gru_nospecialstock').config['common']
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl')
stcok_price = f.loc(['0050'])

eval.regression2Cls_score()
