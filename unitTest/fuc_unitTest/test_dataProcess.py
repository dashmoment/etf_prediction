import sys
sys.path.append('../')
import data_process_list as dp
import numpy as np
import hparam as conf

_dp = dp.train_validation_generaotr()
#f = _dp._load_data('../Data/all_feature_data.pkl')
f = _dp._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl')

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
t_2,v_2 = _dp._split_train_val_side_by_side(stocks_2, 30,5,0.2)
np.random.shuffle(t)

import matplotlib.pyplot as plt
plt.plot(t_2[0,:,3])
plt.plot(t_2[1,:,3])

t_3,v_3 = _dp._split_train_val_side_by_side_random(stocks, 30,5,0.2)

import random
def get_batch_random(data_set, train_step,batch_size, cur_index, feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    sample_step = train_step + 5

    batch = []

    for i in range(batch_size):
        
        rnd = random.randint(0,len(data_set)-sample_step)
        tmpbatch =  np.reshape(data_set[rnd:rnd + sample_step, :], (1, sample_step, -1))
        batch.append(tmpbatch)
    
    batch = np.squeeze(np.array(batch))
    train, label = np.split(batch, [train_step], axis=1)
   
    if feature_size == None: feature_size = np.shape(train)[-1]
    train = train[:,:,:feature_size]
    label = label[:,:,0]

    return train, label


c = conf.config('baseline_random').config['common']

epoch = 0
batch_size = 32
while epoch <  100:

    epoch += 1
    if  c['sample_type'] != 'random': np.random.shuffle(t_3)
 
    #Cehck variable reused
#                tvars = tf.trainable_variables()
#                tvars_vals = sess.run(tvars)
#                for var, val in zip(tvars, tvars_vals):
#                    print(var.name)
#                break

    for i in range(1000):
        batch_index = i*batch_size
        train_data, train_label = get_batch_random(t_3, 10, batch_size, batch_index,  None)

  