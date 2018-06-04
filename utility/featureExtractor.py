import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prepro
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd

from utility import dataProcess as dp
from utility import featureExtractor as f_extr

class feature_extractor:
    def __init__(self, featurelist, data):
        self.featurelist = featurelist
        self.data = data
        
    def rsi(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'RSI' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def kdj(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'KDJ' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def ratio(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'ratio' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def macd(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'MACD' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
    def ud(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'UD' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask

    def velocity(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'velocity' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask

    def ma(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'MA_' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask

    def lag(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'Lag' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask

    def cont(self):
        featuremask = [i for i in range(len(self.featurelist)) if 'cont' in self.featurelist[i]]
        features = gather_features(self.data, featuremask)
        return features, featuremask
    
def create_velocity(stock, meta):

    fe = f_extr.feature_extractor(meta, stock)
    f_velocity, _ = getattr(fe, 'ratio')()
    velocity = f_velocity[1:] - f_velocity[:-1]
    stock = stock[1:]        
    stock = np.concatenate((velocity, stock), axis=1)
    meta_v = ['velocity_1', 'velocity_v2','velocity_v3', 'velocity_v4', 'velocity_v5'] + meta       

    return stock, meta_v


def create_ud_info(stock, meta, backDay = 1): 

    fe = f_extr.feature_extractor(meta, stock)
    f_ud, _ = getattr(fe, 'ud')()
    stock_ud = np.zeros((len(stock), backDay*3))

    f_ud = np.argmax(f_ud, axis = 1)

    for i in range(backDay, len(stock)):
        for j in range(backDay):
            if f_ud[i] == f_ud[i-j-1] == 0:
                stock_ud[i][j*3] = 1

            elif f_ud[i] == f_ud[i-j-1] == 1:
                stock_ud[i][j*3 + 1] = 1

            elif f_ud[i] == f_ud[i-j-1] == 2:
                stock_ud[i][j*3 + 2] = 1

    stock_ud = stock_ud[backDay:]
    stock = stock[backDay:]
    stock = np.concatenate((stock_ud, stock), axis=1)
   
    meta_temp = ['downLag_','fairLag_', 'upLag_'] 
    meta_label = [meta_l + str(bd) for bd in range(backDay) for meta_l in meta_temp]
    meta_ud = meta_label + meta

    return stock, meta_ud


def create_ud_cont(stock, meta): 

    def check_cont_status(stock_ud, index):

        count = [0,0,0]
        for i in range(index-1, 0, -1):
            if stock_ud[i] == stock_ud[index] == 0: count[0]+= 1
            elif stock_ud[i] == stock_ud[index] == 1: count[1]+= 1
            elif stock_ud[i] == stock_ud[index] == 2: count[2]+= 1
            else: break

        return count


    fe = f_extr.feature_extractor(meta, stock)
    f_ud, _ = getattr(fe, 'ud')()
    f_ud = np.argmax(f_ud, axis = 1)

    stock_ud = []
    for i in range(len(stock)):
        stock_ud.append(check_cont_status(f_ud, i))
        

    stock_ud = np.vstack(stock_ud)
    stock = np.concatenate((stock_ud, stock), axis=1)
   
    meta_up = ['contdown','contfair', 'contup'] 
    meta_ud = meta_up + meta

    return stock, meta_ud

def create_ud_cont_2cls(stock, meta): 

    def check_cont_status(stock_ud, index):

        count = [0,0]
        for i in range(index-1, -1, -1):

            if stock_ud[i] == stock_ud[index] == 0: count[0]+= 1
            elif stock_ud[i] == stock_ud[index] == 1 or stock_ud[i] == stock_ud[index] == 2: 
                count[1]+= 1
            else: break

        return count


    fe = f_extr.feature_extractor(meta, stock)
    f_ud, _ = getattr(fe, 'ud')()
    f_ud = np.argmax(f_ud, axis = 1)

    stock_ud = []
    for i in range(len(stock)):
        stock_ud.append(check_cont_status(f_ud, i))
        

    stock_ud = np.vstack(stock_ud)
    stock = np.concatenate((stock_ud, stock), axis=1)
   
    meta_up = ['contdown', 'contup'] 
    meta_ud = meta_up + meta

    return stock, meta_ud

    
def gather_features(data, feature_mask):

    mask = np.zeros(np.shape(data)[-1], dtype= bool)

    for i in range(len(mask)):
        if i in feature_mask:
            mask[i] = True
        else:
            mask[i] = False

    return data[:,mask]