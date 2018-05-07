import sys
sys.path.append('../')
import data_process_list as dp
import numpy as np

_dp = dp.train_validation_generaotr()
f = _dp._load_data('../Data/all_feature_data.pkl')

import pandas as pd
df_dict = { 'ID': [str(1), str(2)]
            }
rng = pd.date_range('1/1/2011', periods=200, freq='D')

idx = 1
for d in rng:
   df_dict[d] = [[idx,idx], [idx,idx]]
   idx += 1

df = pd.DataFrame(df_dict)
df  = df.set_index('ID')

stocks = _dp._selectData2array(df, ['1'], None)
t, v = _dp._split_train_val_side_by_side(stocks, 10, 5, 0.2)
stocks_2 = _dp._selectData2array(f, ['1101'], ['20130102', '20140811'])
np.random.shuffle(t)