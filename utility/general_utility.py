import pickle
import numpy as np

def read_metafile(filepath):
    
    f = open(filepath, 'rb')
    stockList = pickle.load(f)
    pickle.load(f)
    period = pickle.load(f)
    feature_names = pickle.load(f)
    close_price_mean_var = pickle.load(f)
    
    return close_price_mean_var, stockList, period, feature_names

def read_datefile(filepath):
    
    f = open(filepath, 'rb')
    date = pickle.load(f)
   
    return date

def map_ud(predict):

    mapper = {0:-1, 1:0, 2:1}

    return mapper[predict]

def get_sample_weight(label):
    
    weight = []
    
    for i in range(len(label)):
        
        if i < len(label)*0.2:
            weight.append(4)
        elif i >= len(label)*0.2 and i < len(label)*0.4:
            weight.append(2.5)
        elif i >= len(label)*0.4 and i < len(label)*0.6:
            weight.append(2.0)
        elif i >= len(label)*0.6 and i < len(label)*0.8:
            weight.append(1.5)
        else:
            weight.append(1)
            
    return np.flip(weight, axis=0)