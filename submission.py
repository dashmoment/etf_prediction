import pandas as pd
import numpy as np

def map_ud(softmax_output):
    
    ud_meta = {0:-1, 1:0, 2:1}
    ud_index = np.argmax(softmax_output, axis=-1)
    ud = [ud_meta[v] for v in ud_index]
    
    return ud
    
columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
          'Wed_ud', 'Wed_cprice	', 'Thu_ud', 'Thu_cprice', 'Fri_ud',	'Fri_cprice']

df = pd.DataFrame(columns=columns)

ETFid = 50
price = np.array([11,12,13,14,15])
ud_softmax = np.array([[1,0,0],
               [0,1,0],
               [1,0,0],
               [0,0,1],
               [0,1,0]])

ud = np.argmax(ud_softmax, axis=-1)
ud_meta = {0:-1, 1:0, 2:1}
ud_map = [ud_meta[v] for v in ud]
ud_map2 = map_ud(ud_softmax)

results = [50]
for i in range(5):
    results.append(ud[i])
    results.append(price[i])
    
df.loc[0] = results
df = df.set_index('ETFid')

df.to_csv('sample_submit.csv', sep=',')