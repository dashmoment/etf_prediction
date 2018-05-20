import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
from utility import general_utility as gu
import model_zoo as mz
import loss_func as l
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from utility_trial import *
import pandas as pd

def change_raw_columns_name(inputfile:pd.DataFrame):
    inputfile.columns = ["ID", "Date", "name", "open_price", "max", "min", "close_price", "trade"]
    
data_path = '/home/ubuntu/dataset/etf_prediction/raw_data/tetfp.csv'
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

# Group the data by IDs

tasharep_ID_group = tasharep.groupby("ID")
idx = tasharep_ID_group.groups['0050']
s = tasharep.iloc[idx]

stock_price = s.close_price[-10:]

stock_0050_train = stock_price[:-5]
stock_0050_test = stock_price[-5:]


weighted_array = [0.1,0.15,0.2,0.25,0.3]

stock_0050_predict = np.mean(stock_0050_train)

abs_loss = np.abs(stock_0050_predict-stock_0050_test)
score_ref = ((stock_0050_test -abs_loss)/stock_0050_test)*0.5
score = np.sum(score_ref*weighted_array)

scroe_list = {}
for i in range(len(tasharep_ID)):
    
    idx = tasharep_ID_group.groups[tasharep_ID[i]]
    s = tasharep.iloc[idx]
    stock_price = s.close_price[-10:]
    train = stock_price[:-5]
    test = stock_price[-5:]
    
    predict = np.mean(train)
    
    abs_loss = np.abs(predict-test)
    score_ref = ((test -abs_loss)/test)*0.5
    scroe_list[tasharep_ID[i]] = (np.sum(score_ref*weighted_array))

avg_score = []
for k in scroe_list:
    avg_score.append(scroe_list[k])
    
avg_score = np.mean(avg_score)





