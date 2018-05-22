import sys
sys.path.append('../')
import pandas as pd
import pickle

#**********Write to submit file********************
    

with open('./20180520/predict_price_mean.pkl', 'rb') as handle:
    price_list = pickle.load(handle)

with open('./20180520/predict_ud_dow.pkl', 'rb') as handle:
    predict_cls = pickle.load(handle)

columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
          'Wed_ud', 'Wed_cprice ', 'Thu_ud', 'Thu_cprice', 'Fri_ud',    'Fri_cprice']

df = pd.DataFrame(columns=columns)  
idx = 0

for s in price_list.keys():  
    results = [s]
    for i in range(5):
        results.append(predict_cls[s][i])
        results.append(price_list[s][i])
        
    
    df.loc[idx] = results
    idx+=1

df = df.set_index('ETFid') 
df.to_csv('./submit_20180520_v2.csv', sep=',')