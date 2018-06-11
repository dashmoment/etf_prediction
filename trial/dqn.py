import sys
sys.path.append('../')

from tqdm import tqdm
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
from keras.models import Sequential,clone_model, load_model
from keras.layers import Dense
from keras.optimizers import SGD , Adam
import random


from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp



class dqn:
    
    def __init__(self, epsilon, min_epsilon, GAMMA, learning_rate, batchSize, 
                 update_qmodel, model_path = '', isReload = False):
        
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decayrate = (epsilon - self.min_epsilon)/20
        
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.loss = 0
        self.update_qmodel = update_qmodel
        self.model_path = model_path
        
        if not isReload:
            self.model = self.build_model()         
        elif isReload and model_path != '':
            self.model = load_model(model_path)
            
        self.model_q = clone_model(self.model)
            
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=290, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(3, activation='linear'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        
        return model
    
    def action(self, state, istrain = True):
        
        action = np.zeros(3)
        if  random.random() <= self.epsilon and istrain: 
            action[random.randrange(3)] = 1
        else:       
            action[np.argmax(self.model.predict(state))] = 1
                  
        return action
    
    def train(self, memory, iteration): #memory: [ s, a, r , s']
        
         if self.epsilon >  self.min_epsilon:
             if iteration%1000 == 0:
                 self.epsilon = self.epsilon - self.epsilon_decayrate
             
        
         minibatch = random.sample(memory, self.batchSize)
         state = [m[0] for m in minibatch]
         action = [m[1] for m in minibatch]
         reward = [m[2] for m in minibatch]
         state_t1 = [m[3] for m in minibatch]
         
         state = np.vstack(state)
         targets = self.model.predict(state)
         
         
         for i in range(len(minibatch)):
             
             if len(state_t1[i]) == 0:
                 targets[i, action[i]] = reward[i]
             else:
                 predict_t1 = self.model.predict(np.reshape(state_t1[i], (1,-1))) 
                 targets[i, action[i]] = reward[i] + self.GAMMA * np.max(predict_t1, axis=-1)
                 
         self.loss = self.model.train_on_batch(state, targets)
         
         if iteration%self.update_qmodel == 0: 
             self.model_q = clone_model(self.model)
             if self.model_path != '':
                 self.model.save(self.model_path)
                 print('Update Q network and save model')
                 

stock_list =  [
                '0050', '0051',  '0052', '0053', 
                '0054', '0055', '0056', '0057', 
                '0058', '0059', '006201', '006203', 
                '006204', '006208','00690', '00692',  
                '00701', '00713'
              ]


#srcPath = '../Data/0601/all_feature_data_Nm_1_MinMax_120.pkl'
#metaPath = '../Data/0601/all_meta_data_Nm_1_MinMax_120.pkl'
srcPath = '/home/ubuntu/dataset/etf_prediction/0608/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath =  '/home/ubuntu/dataset/etf_prediction/0608/all_meta_data_Nm_1_MinMax_120.pkl'

predict_day = 1
consider_lagday = 5

*_,meta = gu.read_metafile(metaPath)
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

mFeatures = []
mlabels = []
mFeatures_test = []
mlabels_test = []

for s in stock_list:
    single_stock = tv_gen._selectData2array(f, [s], ['20130101', '20180407'])
    single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
    single_stock, meta_ud = f_extr.create_ud_cont(single_stock, meta_v)
    features, label = dp.get_data_from_normal_v2_train(single_stock, meta_ud, predict_day, 
                                                       consider_lagday, 
                                                       ['ratio', 'cont', 'ma', 'velocity'])
    
    mFeatures.append(features)
    mlabels.append(label)


    single_stock_test = tv_gen._selectData2array(f, [s], ['20180407', '20180701'])
    single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
    single_stock_test, meta_ud = f_extr.create_ud_cont(single_stock_test, meta_v)
    features_test, label_test = dp.get_data_from_normal_v2_train(single_stock_test, meta_ud, predict_day, 
                                                                 consider_lagday, 
                                                                 ['ratio', 'cont', 'ma', 'velocity'])

    mFeatures_test.append(features_test)
    mlabels_test.append(label_test)
    
    
features = np.concatenate(mFeatures)  
label = np.concatenate(mlabels)  
features_test = np.concatenate(mFeatures_test)  
label_test = np.concatenate(mlabels_test)
  
#********Play game**********

memory_size = 128
stage_exploration = 128
iteration = 0
total_iteration = 500000
batchSize = 32

memory = []
loss_log = []
reward_log = []

model_path = '/home/ubuntu/model/etf_prediction/dqn/dqn_3lnn.h5'
agent = dqn(epsilon = 0.1, 
            min_epsilon = 0.001,
            GAMMA = 0.99, 
            learning_rate = 1e-4, 
            batchSize = batchSize,
            update_qmodel = 300,
            model_path = model_path, 
            isReload = False)

pbar = tqdm(range(total_iteration))


while iteration < total_iteration:
    
    if (iteration+1)%len(features) == 0:
          state = features[iteration%len(features)]
          label_t = label[iteration%len(features)]
          state_t1 = []
    
    else:
        state = features[iteration%len(features)]
        label_t = label[iteration%len(features)]
        state_t1 = features[(iteration+1)%len(features)]
        
    action = agent.action(np.reshape(state,(1,-1)))   
    reward = np.reshape(np.array(np.equal(np.argmax(action), label_t), np.float32), (1,-1))
    
    if len(memory) >= memory_size:
        memory = memory[1:]
        
    memory.append([state, np.argmax(action, axis=-1), reward, state_t1])
    
    if iteration > stage_exploration:
        agent.train(memory, iteration)
        loss_log.append(agent.loss)
        reward_log.append(reward)     
        pbar.set_description("loss:{}".format(agent.loss))
        
    if iteration%500 == 0:
        
        eval_reward = []
        for i in tqdm(range(len(features_test))):           
            state = features_test[i]
            label_t = label_test[i]
             
            action = agent.action(np.reshape(state,(1,-1)), False)   
            reward = np.reshape(np.array(np.equal(np.argmax(action), label_t), np.float32), (1,-1))
            eval_reward.append(reward)
        print("train reward: ", np.sum(reward_log[-batchSize:])/batchSize)   
        print("Accuracy: ", np.sum(eval_reward)/len(features_test))   

    
    iteration+=1
    pbar.update()
    

plt.figure()
plt.plot(loss_log)
plt.figure()
plt.plot(np.cumsum(reward_log))

#***********DQN Test*************
test_action = []
test_reward = []
for i in tqdm(range(len(features_test))):
   
    state = features_test[i]
    label_t = label_test[i]
     
    action = agent.action(np.reshape(state,(1,-1)), False)   
    reward = np.reshape(np.array(np.equal(np.argmax(action), label_t), np.float32), (1,-1))
    
    test_action.append(action)
    test_reward.append(reward)
    
print("Accuracy: ", np.sum(test_reward)/len(features_test))   












