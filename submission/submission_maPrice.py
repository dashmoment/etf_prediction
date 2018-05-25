import sys
sys.path.append('../')
import numpy as np

from utility import dataProcess as dp
from utility import general_utility as gu
import pandas as pd


tv_gen = dp.train_validation_generaotr()
*_,meta = gu.read_metafile('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm_1_MinMax_94_0050.pkl')
f = tv_gen._load_data('/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_1_MinMax_94_0050.pkl')
data_path = '/home/ubuntu/dataset/etf_prediction/raw_data/tetfp.csv'

#*_,meta = gu.read_metafile('/home/dashmoment/workspace/etf_prediction/Data/all_meta_data_Nm_1_MinMax_94.pkl')
#f = tv_gen._load_data('/home/dashmoment/workspace/etf_prediction/Data/all_feature_data_Nm_1_MinMax_94.pkl')
#data_path = '/home/dashmoment/workspace/etf_prediction/Data/raw_data/20180518/tetfp.csv'

def change_raw_columns_name(inputfile:pd.DataFrame):
    inputfile.columns = ["ID", "Date", "name", "open_price", "max", "min", "close_price", "trade"]
    

tasharep = pd.read_csv(data_path, encoding = "big5-hkscs", dtype=str).dropna(axis = 1) # training data
change_raw_columns_name(tasharep)


tasharep.ID = tasharep.ID.astype(str)
tasharep.Date = tasharep.Date.astype(str)

tasharep["ID"] = tasharep["ID"].astype(str)
tasharep["ID"] = tasharep["ID"].str.strip()

tasharep["Date"] = tasharep["Date"].astype(str)
tasharep["Date"] = tasharep["Date"].str.strip()

tasharep["open_price"] = tasharep["open_price"].astype(str)
tasharep["open_price"] = tasharep["open_price"].str.strip()
tasharep["open_price"] = tasharep["open_price"].str.replace(",", "")
tasharep["open_price"] = tasharep["open_price"].astype(float)

tasharep["max"] = tasharep["max"].astype(str)
tasharep["max"] = tasharep["max"].str.strip()
tasharep["max"] = tasharep["max"].str.replace(",", "")
tasharep["max"] = tasharep["max"].astype(float)

tasharep["min"] = tasharep["min"].astype(str)
tasharep["min"] = tasharep["min"].str.strip()
tasharep["min"] = tasharep["min"].str.replace(",", "")
tasharep["min"] = tasharep["min"].astype(float)

tasharep["close_price"] = tasharep["close_price"].astype(str)
tasharep["close_price"] = tasharep["close_price"].str.strip()
tasharep["close_price"] = tasharep["close_price"].str.replace(",", "")
tasharep["close_price"] = tasharep["close_price"].astype(float)

tasharep["trade"] = tasharep["trade"].astype(str)
tasharep["trade"] = tasharep["trade"].str.strip()
tasharep["trade"] = tasharep["trade"].str.replace(",", "")
tasharep["trade"] = tasharep["trade"].astype(float)

# Get the ID list
tasharep_ID = tasharep.ID.unique()

# Get the Date list
Date = tasharep.Date.unique()
tasharep_ID_group = tasharep.groupby("ID")

price_list = {}
for i in range(len(tasharep_ID)):
    
    idx = tasharep_ID_group.groups[tasharep_ID[i]]
    s = tasharep.iloc[idx]
    stock_price = s.close_price[-5:]
    
    predict = []
    for j in range(5):
        tmp = list(stock_price[j:]) + predict
        predict.append(np.mean(tmp))
    
   
    price_list[tasharep_ID[i]] = predict
    

import pickle
with open('../submission/predict_price_mean.pkl', 'wb') as handle:
    pickle.dump(price_list, handle, protocol=pickle.HIGHEST_PROTOCOL)   