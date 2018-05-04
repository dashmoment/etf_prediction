import pandas as pd
import numpy as np
import hparam as conf
import data_process_list as dp

def map_ud(softmax_output):
    
    ud_meta = {0:-1, 1:0, 2:1}
    ud_index = np.argmax(softmax_output, axis=-1)
    ud = [ud_meta[v] for v in ud_index]
    
    return ud



    
columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
          'Wed_ud', 'Wed_cprice	', 'Thu_ud', 'Thu_cprice', 'Fri_ud',	'Fri_cprice']

stocks = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208', '00690', '00692', '00701', '00713']


df = pd.DataFrame(columns=columns)


c = conf.config('baseline_2in1').config['common']
tv_gen = dp.train_validation_generaotr()
  
eval_set, _ = tv_gen.generate_train_val_set(c['src_file_path'],['1101','1102'], c['input_step'], c['predict_step'], 0.0, None)



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