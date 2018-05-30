import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp


srcPath = '/home/ubuntu/dataset/etf_prediction/0525/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath =  '/home/ubuntu/dataset/etf_prediction/0525/all_meta_data_Nm_1_MinMax_120.pkl'
corrDate_path = '/home/ubuntu/dataset/etf_prediction/0525/xcorr_date_data.pkl'

stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]

stock_list = ['0050']
*_,meta = gu.read_metafile(metaPath) 
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

single_stock = tv_gen._selectData2array(f,stock_list , None)      

#***********Plot close price******************
fig = plt.figure()
fig.suptitle('close price of 0050')
plt.plot(single_stock[:,3])

#***********Plot KDJ_9_9_3******************
fig = plt.figure()
fig.suptitle('KDJ_9_9_3 of 0050')
plt.plot(single_stock[:,84])

#***********Plot RSI_28******************
fig = plt.figure()
fig.suptitle('RSI_28 of 0050')
plt.plot(single_stock[:,82])

close_price = single_stock[:,3]
close_price_df =pd.Series(close_price)


#***********Plot close price_diff******************
fig = plt.figure()
fig.suptitle('close price 1st diff of 0050')
plt.plot(close_price_df.diff(1))


fig = plt.figure()
fig.suptitle('close price 2nd diff of 0050')
plt.plot(close_price_df.diff(2))



fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(close_price_df.diff(1)[-200:],lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(close_price_df.diff(1)[-200:],lags=50, ax=ax2)
