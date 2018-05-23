import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf

import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l
import sklearn.preprocessing as prepro
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd


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
    
    
def gather_features(data, feature_mask):

    mask = np.zeros(np.shape(data)[-1], dtype= bool)

    for i in range(len(mask)):
        if i in feature_mask:
            mask[i] = True
        else:
            mask[i] = False

    return data[:,mask]