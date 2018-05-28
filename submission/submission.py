import sys
sys.path.append('../')
import pandas as pd
import pickle

#**********Write to submit file********************
    

with open('./20180525/predict_price_mean.pkl', 'rb') as handle:
    predict_price = pickle.load(handle)

with open('./20180525/predict_ud_xgb_speicalDate_nsw_npw_cscore.pkl', 'rb') as handle:
    predict_ud = pickle.load(handle)

columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
          'Wed_ud', 'Wed_cprice ', 'Thu_ud', 'Thu_cprice', 'Fri_ud',    'Fri_cprice']

df = pd.DataFrame(columns=columns)  
idx = 0

for s in predict_price.keys():  
    results = [s]
    for i in range(5):
        results.append(predict_ud[s][i])
        results.append(predict_price[s][i])
        
    
    df.loc[idx] = results
    idx+=1

df = df.set_index('ETFid') 
df.to_csv('./submit_20180525_xgb_speicalDate_nsw_npw_cscore.csv', sep=',')