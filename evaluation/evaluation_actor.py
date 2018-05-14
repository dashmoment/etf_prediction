import sys
sys.path.append('../')
import numpy as np

import hparam as conf
import evaluation_zoo as evalf
from utility import general_utility

conf_reg = conf.config('test_onlyEnc_biderect_gru_nospecialstock').config['common']
close_price_mean_var, *_ = general_utility.read_metafile(conf_reg['meta_file_path'])

mean = close_price_mean_var.mean_[0]
std = np.sqrt(close_price_mean_var.var_[0])
stockID = ['0050']

reg = evalf.regression_score(conf_reg,stockID, mean, std)
reg_score, *_ = reg.regression_score()

r2Cls = evalf.regression2Cls_score(conf_reg, stockID, mean, std)
r2Cls_predict, *_ = r2Cls.regression2Cls_score()

conf_cls = conf.config('test_onlyEnc_biderect_gru_nospecialstock_cls').config['common']
cls = evalf.classification_score(conf_cls, stockID)
cls_score, *_ = cls.classification_score()
