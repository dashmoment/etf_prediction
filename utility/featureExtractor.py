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
    
def create_velocity(stock, meta):

    fe = f_extr.feature_extractor(meta, stock)
    f_velocity, _ = getattr(fe, 'ratio')()
    velocity = f_velocity[1:] - f_velocity[:-1]
    stock = stock[1:]        
    stock = np.concatenate((velocity, stock), axis=1)
    meta_v = ['velocity_1', 'velocity_v2','velocity_v3', 'velocity_v4', 'velocity_v5'] + meta       

    return stock, meta_v

    
def gather_features(data, feature_mask):

    mask = np.zeros(np.shape(data)[-1], dtype= bool)

    for i in range(len(mask)):
        if i in feature_mask:
            mask[i] = True
        else:
            mask[i] = False

    return data[:,mask]