import sys
sys.path.append('../')
sys.path.append('../../')
import evaluation_2model as eval
import hparam as conf
import data_process_list as dp
import numpy as np

tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_0_89.pkl')

stcok_price = f.iloc[0]



def mapfuc(x):
                
    if x < 0: return 0
    elif x > 0: return 2
    else: return 1
price = []
price_updown = []
for i in range(1, len(stcok_price)):
    price.append(stcok_price.iloc[i][3] - stcok_price.iloc[i-1][3])
    price_updown.append(mapfuc(price[-1]))   
updown = []
for i in range(1, len(stcok_price)):
    updown.append(np.argmax(stcok_price.iloc[i][-3:], axis=-1))


print("[Test] Price Label is equal to up-down: ", np.equal(price_updown,updown).all())