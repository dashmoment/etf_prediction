import numpy as np
import pandas as pd
import _pickle as pickle
from tqdm import tqdm



verbose_state = True


def print_c(print_content, verbose = verbose_state):
    
    if verbose == True:
        print(print_content)
        
class train_validation_generaotr:
    
     #load weiyu provide data
    def _load_data(self, filepath):
        
        print_c("Read pickle data")
        
        process_data = pd.read_pickle(filepath)
    
        print_c("Finish read pickle data")
        
        return process_data
    
    #Select stock and return as np.array
    def _selectData2array(self, data, stock_IDs, time_period):
        #data: processed data
        #stock_IDs: [stock1,stock2,...]
        #time_period: [start_date:str, end_data:str]
        
        #return: len(stock_IDs) == 1: [data_size, feature_size]
        #        len(stock_IDs) > 1: [data_size, feature_size, stocks]   
    
        print_c("Process data for stock:{}".format(stock_IDs))
    
        #select time period
        if time_period != None:
            mask = (data.columns >= time_period[0]) & (data.columns < time_period[1])
            data = data.iloc[:,mask]
                
        
        stock = data.loc[stock_IDs]
#        stock = stock.dropna(axis=1)
        stock = np.hstack(np.array(stock))
        stock = np.vstack(stock)

        if len(stock_IDs) > 1:
            print(np.shape(stock))
            stock = np.split(stock, len(stock_IDs))
            stock =np.dstack(stock)
        
        return stock
    
    def _split_train_val_side_by_side(self, data, train_windows, predict_windows, train_val_ratio):
        
        print_c('Split train and validation data from {} data'.format(len(data)))   
        sample_window =  train_windows + predict_windows 
        total_len = len(data)
        pivot = int(total_len*(1-train_val_ratio))
        train_data = data[:pivot]
        valid_data = data[pivot:]
        
        train = []
        validataion = []
        
        for i in range(len(train_data)-sample_window):
            train.append(train_data[i:i+sample_window])
            
        for i in range(len(valid_data)-sample_window):
            validataion.append(valid_data[i:i+sample_window])
            
        if len(train) > 0: train = np.stack(train)
        if len(validataion) > 0: validataion = np.stack(validataion)
        
        return train, validataion
            
        
        
    def _split_train_val(self, data, train_windows, predict_windows, train_val_ratio):
        
        #train_windows: windows size for model input data
        #predict_windows: windows size for model output data
        #train_val_ratio: ration of validation_set_size/training_set_size
        
        #return: train, validation set: [set_size, total_time_windows, feature_size]
        
        print_c('Split train and validation data from {} data'.format(len(data)))
        
        sample_window =  train_windows + predict_windows 
        total_len = len(data)
        cut_len = total_len - sample_window  
        
        train = []
        validataion = []
        
        idx = 0
        pivot = 0
        N_train = cut_len//(1+2*predict_windows*train_val_ratio)
        N_val = (cut_len - N_train)//(2*predict_windows)
        
       
        if N_val < 1: switch_pivot = 0.1
        else: switch_pivot = N_train//N_val

        
        pbar_s = tqdm(cut_len)
   
        while idx < cut_len:
            
            if pivot == switch_pivot:      
                pivot = 0
                idx += predict_windows - 1
                pbar_s.update(predict_windows - 1)
                
                if total_len < (idx + sample_window): break
                validataion.append(data[idx:idx+sample_window])
                idx += predict_windows
                pbar_s.update(predict_windows)
            else:
                train.append(data[idx:idx+sample_window])
                pivot += 1
                idx += 1       
                pbar_s.update(1)
                
        if len(train) > 0: train = np.stack(train)
        if len(validataion) > 0: validataion = np.stack(validataion)
    
        return train, validataion
        
    
    def generate_train_val_set(self, filepath, stock_IDs, train_windows, predict_windows,
                               train_val_ratio, time_period = None):
        
        process_data = self._load_data(filepath)
        stock_data = self._selectData2array(process_data, stock_IDs, time_period)
        #train, valid = self._split_train_val(stock_data, train_windows, predict_windows, train_val_ratio)
        train, valid = self._split_train_val_side_by_side(stock_data, train_windows, predict_windows, train_val_ratio)
        return train, valid
    
    def get_test_data(self, train_windows, stocks):
        
        train_dict = dict()
             
        for s in stocks:
            train_dict[s] = stocks[s][-train_windows:]
    
        return train_dict
    
    def generate_test_set(self, filepath, stock_IDs, train_windows):
        
        testSet = {}
        process_data = self._load_data(filepath)
        
        for s in stock_IDs:
            d = self._selectData2array(process_data, [s], None)
            testSet[s] = d
            
        return self.get_test_data(train_windows, testSet)
    

#Simple Demo
#filepath = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
#tv_gen = train_validation_generaotr()
#f = pd.read_pickle(filepath)
#stocks = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208', '00690', '00692', '00701', '00713']

#testSet = tv_gen.generate_test_set(filepath, stocks, 30)




        




        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    