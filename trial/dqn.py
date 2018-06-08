import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD , Adam
import random


from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp



class dqn:
    
    def __init__(self, epsilon, GAMMA, learning_rate, batchSize):
        
        self.epsilon = epsilon
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.model = self.build_model()
        self.loss = 0
        
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=290, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(3, activation='linear'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        
        return model
    
    def action(self, state):
        action = np.zeros(3)
        if  random.random() <= self.epsilon:
    
            action[random.randrange(3)] = 1
        else:
            
            action[np.argmax(self.model.predict(state))] = 1
                  
        return action
    
    def train(self, memory): #memory: [ s, a, r , s']
         minibatch = random.sample(memory, self.batchSize)
         state = [m[0] for m in minibatch]
         action = [m[1] for m in minibatch]
         reward = [m[2] for m in minibatch]
         state_t1 = [m[3] for m in minibatch]
         
         targets = self.model.predict(state)
         predict_t1 = self.model.predict(state_t1) 
         
         for i in range(len(minibatch)):
             
             if state_t1 == []:
                 targets[i, action[i]] = reward[i]
             else:
                 targets[i, action[i]] = reward[i] + self.GAMMA * np.max(predict_t1[i])
                 
         self.loss += self.model.train_on_batch(state, targets)


stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]


srcPath = '../Data/0601/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath = '../Data/0601/all_meta_data_Nm_1_MinMax_120.pkl'

*_,meta = gu.read_metafile(metaPath)
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

single_stock = tv_gen._selectData2array(f, ['0050'], None)
single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
single_stock, meta_ud = f_extr.create_ud_cont(single_stock, meta_v)
features, label = dp.get_data_from_normal_v2_train(single_stock, meta_ud, 1, 5, ['ratio', 'cont', 'ma', 'velocity'])

#********Play game**********

memory_size = 80
stage_exploration = 80
iteration = 0
total_iteration = 1000

memory = []

agent = dqn(epsilon = 0.05, 
            GAMMA = 0.1, 
            learning_rate = 1e-4, 
            batchSize = 32)

while iteration < total_iteration:
    
    state = features[iteration%len(features)]
    label_t = label[iteration%len(features)]
    
    iteration+=1
    
    action = agent.action(np.reshape(state,(1,-1)))
    
    if len(memory) > memory_size:
        memory = memory[1:]
    
    
    
    
















