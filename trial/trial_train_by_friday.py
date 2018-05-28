import sys
sys.path.append('../')

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, classification_report

from utility import general_utility as gu
from utility import featureExtractor as f_extr
from utility import dataProcess as dp

def get_period_raw_data(raw, period, shift = 1):

    mask = (raw.columns >= period[0]) & (raw.columns < period[1])
    raw_prediod = raw.iloc[:,mask]   

    if shift > 0:
          raw_prediod =  raw_prediod.iloc[:,1:] #Due to get velocity

    return raw_prediod

stock_list = ['0051']

srcPath = '/home/ubuntu/dataset/etf_prediction/0525/all_feature_data_Nm_1_MinMax_120.pkl'
metaPath =  '/home/ubuntu/dataset/etf_prediction/0525/all_meta_data_Nm_1_MinMax_120.pkl'

*_,meta = gu.read_metafile(metaPath)
tv_gen = dp.train_validation_generaotr()
f = tv_gen._load_data(srcPath)

period = ('20130414','20180414')

raw = f  
#mask = (raw.columns >= period[0]) & (raw.columns < period[1])   
#raw_train = raw.iloc[:,mask]

feature_list =  ['ratio']
single_stock = tv_gen._selectData2array(f, ['0051'], period)
single_stock, meta_v = f_extr.create_velocity(single_stock, meta)
#raw_train = raw_train.iloc[:,1:]


raw_train =  get_period_raw_data(raw, period)

stocks = dp.clean_stock(single_stock,meta_v,feature_list)   
 
df = pd.DataFrame({'date':raw_train.columns})
df['date'] = pd.to_datetime(df['date'])
df['dow'] = df['date'].dt.dayofweek
dow_array = np.array(df['dow'])
dow_array_mask_mon =  np.equal(dow_array, 4)

print('*****************************')
print(np.array(df['date'][-len(stocks):])[-1])

predict_day = 0

def get_mask(dow_array_mask_mon):
         for i in range(5):
             dow_array_mask_mon[i] = False
         
         dow_array_mask = [dow_array_mask_mon]
         for j in range(1, 5):
             tmp_mask = np.zeros(np.shape(dow_array_mask_mon), np.bool)
             for i in range(1, len(dow_array_mask_mon)):
                if dow_array_mask_mon[i] == True: 
                    tmp_mask[i-j] = True              
                else: 
                    tmp_mask[i] = False
             dow_array_mask.append(tmp_mask)
         return dow_array_mask
     
dow_array_mask = get_mask(dow_array_mask_mon)
dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}
features = {}

isShift = True
for d in range(5):
        features[dow[d]] = {}
        shifted_stock = stocks[dow_array_mask[d]]

        if isShift == True: shifted_stock = shifted_stock[:-1]      
        fe = f_extr.feature_extractor(meta_v, shifted_stock)
        
        for feature_name in feature_list:
            features[dow[d]][feature_name], _ = getattr(fe, feature_name)()
feature_concat = []     
dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}    
for i in range(5):
        for k in  features[dow[i]]:
            feature_concat.append( features[dow[i]][k])          
train = np.concatenate(feature_concat, axis=1)
            
label = np.argmax(stocks[dow_array_mask[predict_day]][1:, -3:], axis=-1)


#***********Test****************
period_test = ('20180414','20180610')

#mask = (raw.columns >= period_test[0]) & (raw.columns < period_test[1])   
#raw_test = raw.iloc[:,mask]


single_stock_test = tv_gen._selectData2array(f, ['0051'], period_test)
single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
stocks_test = dp.clean_stock(single_stock_test,meta_v,feature_list)  

raw_test  =  get_period_raw_data(raw, period_test)

#raw_test = raw_test.iloc[:,1:]


  
df = pd.DataFrame({'date':raw_test.columns})
df['date'] = pd.to_datetime(df['date'])
df['dow'] = df['date'].dt.dayofweek
dow_array = np.array(df['dow'])
dow_array_mask_mon_test =  np.equal(dow_array, 4)
dow_array_mask_test = get_mask(dow_array_mask_mon_test)
print('*****************************')
print(np.array(df['date'][-len(stocks):])[-1])
dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}
features = {}

isShift = True
for d in range(5):
        features[dow[d]] = {}
        shifted_stock = stocks_test[dow_array_mask_test[d]]

        if isShift == True: shifted_stock = shifted_stock[:-1]      
        fe = f_extr.feature_extractor(meta_v, shifted_stock)
        
        for feature_name in feature_list:
            features[dow[d]][feature_name], _ = getattr(fe, feature_name)()
feature_concat = []     
dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}    
for i in range(5):
        for k in  features[dow[i]]:
            feature_concat.append( features[dow[i]][k])          
test = np.concatenate(feature_concat, axis=1)
label_test = np.argmax(stocks_test[dow_array_mask_test[predict_day]][1:, -3:], axis=-1)
            



model = xgb.XGBClassifier( max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True, 
                                            objective='multi:softmax', num_class=3
                                           )


score = np.mean(cross_val_score(model, train, label, cv=3,
                                    n_jobs = 5, 
                                    #scoring= scoreF.time_discriminator_score,
                                    #fit_params = sample_weight
                                    ))
model.fit(train, label)
y_xgb_train = model.predict(train)
y_xgb_test = model.predict(test)
print("Train Accuracy of day {} [DOW][{}]: {}".format(4,'xgb', accuracy_score(label, y_xgb_train)))
print("Validation Accuracy  {} [DOW][{}]: {} ".format(4, 'xgb', accuracy_score(label_test, y_xgb_test)))


#*******************************************************************************************
#features, label = dp.get_data_from_dow(f , single_stock, meta_v, predict_day, feature_list)
single_stock = tv_gen._selectData2array(f, ['0051'], period)
single_stock, meta_v = f_extr.create_velocity(single_stock, meta)                    
raw_train_c = get_period_raw_data(f, period)

features_c, label = dp.get_data_from_dow_friday(raw_train_c , single_stock, meta_v, predict_day, feature_list)

feature_concat = []
dow = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri'}

for i in range(5):
    for k in  features[dow[i]]:
        feature_concat.append( features_c[dow[i]][k])          
data_feature = np.concatenate(feature_concat, axis=1)
   
train_val_set_days = {'train': data_feature,
                      'train_label': label}


train_data = train_val_set_days['train']
train_label = train_val_set_days['train_label']

#*****************************************************

test_period = ['20180414','20180601']             
single_stock_test = tv_gen._selectData2array(f, ['0051'], test_period)
single_stock_test, meta_v = f_extr.create_velocity(single_stock_test, meta)
raw_test = get_period_raw_data(f, test_period)
features_test, label_test = dp.get_data_from_dow_friday(raw_test, single_stock_test, meta_v, predict_day, feature_list)

feature_concat_test = []                   
for i in range(5):
    for k in  features_test[dow[i]]:
        feature_concat_test.append(features_test[dow[i]][k])


data_feature_test = np.concatenate(feature_concat_test, axis=1)                   
test_val_set_days = {'test': data_feature_test,
                      'test_label': label_test}

test_data = test_val_set_days['test']
test_label = test_val_set_days['test_label']

model = xgb.XGBClassifier( max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True, 
                                            objective='multi:softmax', num_class=3
                                           )


score = np.mean(cross_val_score(model, train, label, cv=3,
                                    n_jobs = 5, 
                                    #scoring= scoreF.time_discriminator_score,
                                    #fit_params = sample_weight
                                    ))
model.fit(train_data, train_label)
y_xgb_train = model.predict(train_data)
y_xgb_test = model.predict(test_data)
print("Train Accuracy of day {} [DOW][{}]: {}".format(4,'xgb', accuracy_score(train_label, y_xgb_train)))
print("Validation Accuracy  {} [DOW][{}]: {} ".format(4, 'xgb', accuracy_score(test_label, y_xgb_test)))
