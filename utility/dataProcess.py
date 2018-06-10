import numpy as np
import pandas as pd
from tqdm import tqdm
from utility import general_utility as ut
from utility import featureExtractor as fe_extr


verbose_state = True

def get_data_by_date(raw, startDay, period):

    index = [idx for idx in range(len(raw.columns)) if raw.columns[idx] == startDay]
    period_list = [raw.columns[idx] for idx in range(index[0], index[0]+period)]
    data_in_period = raw[period_list]
    
    return data_in_period

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
    
        #print_c("Process data for stock:{}".format(stock_IDs))
    
        #select time period
        if time_period != None:
            mask = (data.columns >= time_period[0]) & (data.columns < time_period[1])
            data = data.iloc[:,mask]
                       
        stock = data.loc[stock_IDs]
        stock = np.hstack(np.array(stock))
        stock = np.vstack(stock)

        if len(stock_IDs) > 1:
            stock = np.split(stock, len(stock_IDs))
            stock =np.dstack(stock)
        
        return stock

    def _selectData2array_specialDate(self, raw, dates, period, stock_IDs):

        data_by_dates = []
        for d in dates:
            data_by_dates.append(get_data_by_date(raw, d, period))
            
        data_by_dates = pd.concat(data_by_dates, axis=1)
        stock = data_by_dates.loc[stock_IDs]
        stock = np.vstack(stock)

        return stock
    
    def _selectData2array_specialDate_v2(self, raw, dates, corr_date, period, stock_IDs):

        data_by_dates = []
        for d in dates:
            data_by_dates.append(get_data_by_date(raw, d, period))
            
        data_by_dates = pd.concat(data_by_dates, axis=1)
        stock = data_by_dates.loc[stock_IDs]
        stock = np.vstack(stock)
        
        stock = np.reshape(stock, (corr_date, period, -1))
        return stock

    
    def _split_train_val_side_by_side(self, data, train_windows, predict_windows, train_val_ratio):
        
        print_c('Split train and validation data from {} data'.format(len(data)))   
        sample_window =  train_windows + predict_windows 
        total_len = len(data)
        pivot = int(total_len*(1-train_val_ratio))
        train_data = data[:pivot]
        valid_data = data[pivot - train_windows:]
        
        train = []
        validataion = []
        
        for i in range(len(train_data)-sample_window):
            train.append(train_data[i:i+sample_window])
            
        for i in range(len(valid_data)-sample_window):
            validataion.append(valid_data[i:i+sample_window])
            
        if len(train) > 0: train = np.stack(train)
        if len(validataion) > 0: validataion = np.stack(validataion)
        
        return train, validataion

    def _split_train_val_side_by_side_random(self, data, train_windows, predict_windows, train_val_ratio):
        
        print_c('Split train and validation data from {} data'.format(len(data)))   
        total_len = len(data)
        pivot = int(total_len*(1-train_val_ratio))
        train_data = data[:pivot]
        valid_data = data[pivot:]
        
        train = train_data
        validataion = valid_data
        
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

    def generate_train_val_set_random(self, filepath, stock_IDs, train_windows, predict_windows,
                               train_val_ratio, time_period = None):
        
        process_data = self._load_data(filepath)
        stock_data = self._selectData2array(process_data, stock_IDs, time_period)

        #train, valid = self._split_train_val(stock_data, train_windows, predict_windows, train_val_ratio)
        if len(stock_IDs) > 1:
            train, valid = self._split_train_val_side_by_side(stock_data, train_windows, predict_windows, train_val_ratio)
        else:
            train, valid = self._split_train_val_side_by_side_random(stock_data, train_windows, predict_windows, train_val_ratio)
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


    def generate_train_val_set_mStock(self, filepath, stock_IDs, train_windows, predict_windows, train_val_ratio,
                                      is_special_list = False,  
                                      metafile = '/home/dashmoment/workspace/etf_prediction/Data/all_meta_data_Nm[0]_59.pkl'):

#    train_windows = 50
#    predict_windows = 5
#    train_val_ratio = 0.2
#    filepath = './Data/all_feature_data_Nm[0]_59.pkl'
        
        *_, feature_names = ut.read_metafile(metafile)
          
        testSet = self._load_data(filepath)
        
        clean_stock = {}
        missin_feature = []
        
        if is_special_list:
            
            special_list = {
                            '00690':"20170330", 
                            '00692':"20170516", 
                            '00701':"20170816", 
                            '00713':"20170927"}
            
            for s in special_list:
                
                mask = (testSet.columns > special_list[s]) 
                cut_testSet = testSet.iloc[:,mask]
            
                stock_s = cut_testSet.loc[s]
               
                clean_set = []
                [clean_set.append(row) for row in stock_s]
            
                clean_set = np.vstack(clean_set)
                
                tmpDF = pd.DataFrame(clean_set, columns=feature_names)
                missin_feature.append(tmpDF.columns[tmpDF.isnull().any()].tolist())
                tmpDF = tmpDF.dropna(axis=[1]) 
                clean_stock[s] = tmpDF
        
            all_stock_list = stock_IDs + ["00690", "00692", "00701", "00713"]
        else:
            all_stock_list = stock_IDs

        for s in stock_IDs:
            stock = testSet.loc[s]
            clean_set = []
            [clean_set.append(row) for row in stock]
            clean_set = np.vstack(clean_set)
            tmpDF = pd.DataFrame(clean_set, columns=feature_names)
            if is_special_list: clean_stock[s] = tmpDF.drop(missin_feature[-1], axis=1)  
            else: clean_stock[s] = tmpDF
        
        train = []
        validation = []
        train_raw = {}
        validation_raw = {}
        
        for s in all_stock_list:
            
            tmp_train, tmp_validation = self._split_train_val_side_by_side(clean_stock[s], train_windows, predict_windows, train_val_ratio)
            train.append(tmp_train)
            validation.append(tmp_validation)
            
            train_raw[s] = tmp_train
            validation_raw[s] = tmp_validation
            
        train = np.vstack(train)
        validation = np.vstack(validation)

        return train, validation, train_raw, validation_raw, missin_feature
    

########Simple Demo#############
#import math
#
#filepath = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
#tv_gen = train_validation_generaotr()
#f = pd.read_pickle(filepath)
#stocks = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208']
#process_data = tv_gen._load_data(filepath)
#stock_data = tv_gen._selectData2array(process_data, stocks, None)
#train, valid = tv_gen._split_train_val_side_by_side(stock_data, 50, 5, 0.2)



class data_processor:
    
    def __init__(self, srcPath, lagday = 0, period = None, stockList = None):
        
        self.srcPath = srcPath
        self.lagday = lagday 
        self.period = period 
        self.stockList = stockList
        
    def clean_data(self):
        
        tv_gen = train_validation_generaotr()
        f = tv_gen._load_data(self.srcPath)
        
        if self.stockList == None:
            self.stockList = f.index
        else:
            self.stockList = self.stockList
        
        clean_stock = {}
        
        for s in self.stockList:
            single_stock = tv_gen._selectData2array(f, [s],  self.period)
            tmpStock = []
            
            for i in range(len(single_stock)):
                if not np.isnan(single_stock[i,0:5]).all():
                    tmpStock.append(single_stock[i])
            single_stock = np.array(tmpStock)
            
            if self.lagday > 0:
                data = single_stock[:-self.lagday]
                label = single_stock[self.lagday:, -3:]
            else:
                 data = single_stock
                 label = single_stock[:, -3:]

            clean_stock[s] = {'data': data,
                              'label_ud':label}

            
        return clean_stock
    
    def split_train_val_set(self, data, label, split_ratio):
        
        split_pivot = int(split_ratio*len(data))
        train_val_set = {'train': data[:-split_pivot],
                         'train_label': label[:-split_pivot],
                         'test': data[-split_pivot:],
                         'test_label': label[-split_pivot:]}
        
        return train_val_set
    
    def split_train_val_set_mstock(self, stocks, split_ratio, stock_list = None):
        
        
        single_train_val_set = {}
        
        if stock_list == None: stock_list =  stocks
        
        for k in stock_list:
            stock = stocks[k]
            tmpdata = stock['data']
            tmp_label = stock['label_ud']
            tmp_train_val_set = self.split_train_val_set(tmpdata, tmp_label, split_ratio)   
            single_train_val_set[k] = tmp_train_val_set
        
        train_val_set = {}
        train_val_set['train'] = np.concatenate([single_train_val_set[k]['train'] for k in single_train_val_set], axis=0)
        train_val_set['train_label'] = np.concatenate([single_train_val_set[k]['train_label'] for k in single_train_val_set], axis=0)
        train_val_set['test'] = np.concatenate([single_train_val_set[k]['test'] for k in single_train_val_set], axis=0)
        train_val_set['test_label'] = np.concatenate([single_train_val_set[k]['test_label'] for k in single_train_val_set], axis=0)
        return train_val_set
        
def clean_stock(single_stock, meta, feature_list):

    tmpStock = []

    fe = fe_extr.feature_extractor(meta, single_stock)

    feature_mask = []
    _, tmp_mask = getattr(fe, 'ratio')()
    feature_mask += tmp_mask

    for f in feature_list:
        _, tmp_mask = getattr(fe, f)()
        feature_mask += tmp_mask

    for i in range(len(single_stock)):
        if not np.isnan(single_stock[i,list(set(feature_mask))]).any():
            tmpStock.append(single_stock[i])
    single_stock = np.array(tmpStock)

    return single_stock

def get_data_from_dow(raw, stocks, meta, predict_day, feature_list = ['ratio'], isShift = True):

    stocks = clean_stock(stocks,meta, feature_list)   
    
    df = pd.DataFrame({'date':raw.columns})
    df['date'] = pd.to_datetime(df['date'])
    df['dow'] = df['date'].dt.dayofweek
    dow_array = np.array(df['dow'][-len(stocks):])
    #print('*****************************')
    #print(np.array(df['date'][-len(stocks):])[-1])
    dow_array_mask_mon =  np.equal(dow_array, predict_day)
     
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
    
    for d in range(5):
        features[dow[d]] = {}
        shifted_stock = stocks[dow_array_mask[d]]

        if isShift == True: shifted_stock = shifted_stock[:-1]      
        fe = fe_extr.feature_extractor(meta, shifted_stock)
        
        for feature_name in feature_list:
            features[dow[d]][feature_name], _ = getattr(fe, feature_name)()
            
    if isShift == True: label = np.argmax(stocks[dow_array_mask[0]][1:, -3:], axis=-1)
    else: label = np.argmax(stocks[dow_array_mask[0]][:, -3:], axis=-1)

    return features, label     

def get_data_from_dow_friday(raw, stocks, meta, predict_day, feature_list = ['ratio'], isShift = True):

    stocks = clean_stock(stocks,meta, feature_list)   
    
    df = pd.DataFrame({'date':raw.columns})
    df['date'] = pd.to_datetime(df['date'])
    df['dow'] = df['date'].dt.dayofweek
    dow_array = np.array(df['dow'][-len(stocks):])
    #print('*****************************')
    #print(np.array(df['date'][-len(stocks):])[-1])
    dow_array_mask_mon =  np.equal(dow_array, 4)
     
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
    
    for d in range(5):
        features[dow[d]] = {}
        shifted_stock = stocks[dow_array_mask[d]]

        if isShift == True: shifted_stock = shifted_stock[:-1]      
        fe = fe_extr.feature_extractor(meta, shifted_stock)
        
        for feature_name in feature_list:
            features[dow[d]][feature_name], _ = getattr(fe, feature_name)()
            
    if isShift == True: label = np.argmax(stocks[dow_array_mask[predict_day]][1:, -3:], axis=-1)
    else: label = np.argmax(stocks[dow_array_mask[predict_day]][:, -3:], axis=-1)

    return features, label     

def get_data_from_normal(stocks, meta, predict_day, feature_list = ['ratio'], isShift=True):

    stocks = clean_stock(stocks,meta, feature_list)  
    current_mask =  np.ones(len(stocks), np.bool)
    
    def get_mask(current_mask):
         for i in range(5):
             current_mask[i] = False
         
         shift_array_mask = [current_mask]
         for j in range(1, 5):
             tmp_mask = np.zeros(np.shape(current_mask), np.bool)
             for i in range(1, len(current_mask)):
                if current_mask[i] == True: 
                    tmp_mask[i-j] = True              
                else: 
                    tmp_mask[i] = False
             shift_array_mask.append(tmp_mask)
         return shift_array_mask
    
    mask = get_mask(current_mask)
    
    features = {}
    
    for d in range(5):
        features[d] = {}
        shifted_stock = stocks[mask[d]]
        if isShift == True: shifted_stock = shifted_stock[:-predict_day]
        
        fe = fe_extr.feature_extractor(meta, shifted_stock)
        
        for feature_name in feature_list:
            features[d][feature_name], _ = getattr(fe, feature_name)()
            
    if isShift == True: label = np.argmax(stocks[mask[0]][predict_day:, -3:], axis=-1)
    else: label = np.argmax(stocks[mask[0]][:, -3:], axis=-1)
    
    return features, label

def get_data_from_normal_v2_train(stocks, meta, predict_day, consider_lagday,feature_list = ['ratio'], isShift=True):
    
         idx = len(stocks)
         label = {                
                  1:[],
                  2:[],
                  3:[],
                  4:[],
                  5:[]
                  }
         
         data = {                
                  1:[],
                  2:[],
                  3:[],
                  4:[],
                  5:[]
                  }
         
         while idx > 5:    
             for i in range(1,6):
                 
                 label[6-i].append(np.argmax(stocks[idx-i, -3:],axis=-1))
             if isShift: idx = idx - 5
             
             for i in range(1,6):
                 data[6-i].append(stocks[idx-i])
                 #print(idx-i, ' ',stocks[idx-i][92])
             if not isShift: idx = idx - 5

         features = {}
        
         for d in data.keys():
                features[d] = {}
                data[d] = np.stack(data[d], axis=0)
                fe = fe_extr.feature_extractor(meta, data[d])
            
                for feature_name in feature_list:
                     features[d][feature_name], _ = getattr(fe, feature_name)()
                     
         feature_concat = []
         for i in range(5,5-consider_lagday, -1):
             for k in  features[i]:
                 feature_concat.append( features[i][k])
        
         data_feature = np.concatenate(feature_concat, axis=1)
         data = data_feature
         label = label[predict_day]

         return data, label

def get_data_from_normal_v2_test(stocks, meta, predict_day, model_config, isShift=True):
    
         idx = len(stocks)
         label = {                
                  1:[],
                  2:[],
                  3:[],
                  4:[],
                  5:[]
                  }
         
         data = {                
                  1:[],
                  2:[],
                  3:[],
                  4:[],
                  5:[]
                  }
         
         while idx > 5:    
             for i in range(1,6):
                 
                 label[6-i].append(np.argmax(stocks[idx-i, -3:],axis=-1))
             if isShift: idx = idx - 5
             
             for i in range(1,6):
                 data[6-i].append(stocks[idx-i])
                 #print(idx-i, ' ',stocks[idx-i][92])
             if not isShift: idx = idx - 5

         features = {}
        
         for d in data.keys():
                features[d] = {}
                data[d] = np.stack(data[d], axis=0)
                fe = fe_extr.feature_extractor(meta, data[d])
            
                for feature_name in model_config['features']:
                     features[d][feature_name], _ = getattr(fe, feature_name)()
                     
         feature_concat = []
         for i in range(5,5-model_config['days'], -1):
             for k in  features[i]:
                 feature_concat.append( features[i][k])
        
         data_feature = np.concatenate(feature_concat, axis=1)
         data = data_feature
         label = label[predict_day]

         return data, label

def get_data_from_normal_weekly_train(stocks, meta, consider_lagday,feature_list = ['ratio'], isShift=True):
    
         idx = len(stocks)
         label = {                
                  1:[],
                  2:[],
                  3:[],
                  4:[],
                  5:[]
                  }
         
         data = {                
                  1:[],
                  2:[],
                  3:[],
                  4:[],
                  5:[]
                  }
         
         while idx > 5:    
             for i in range(1,6):
                 
                 label[6-i].append(np.argmax(stocks[idx-i, -3:],axis=-1))
             if isShift: idx = idx - 5
             
             for i in range(1,6):
                 data[6-i].append(stocks[idx-i])
                 #print(idx-i, ' ',stocks[idx-i][92])
             if not isShift: idx = idx - 5

         features = {}
        
         for d in data.keys():
                features[d] = {}
                data[d] = np.stack(data[d], axis=0)
                fe = fe_extr.feature_extractor(meta, data[d])
            
                for feature_name in feature_list:
                     features[d][feature_name], _ = getattr(fe, feature_name)()
                     
         feature_concat = []
         for i in range(5,5-consider_lagday, -1):
             for k in  features[i]:
                 feature_concat.append( features[i][k])
        
         data_feature = np.concatenate(feature_concat, axis=1)
         data = data_feature

         weekly_label = []
         for i in range(1, 6):
            weekly_label.append([ut.map_ud(_label) for _label in label[i]])

         weekly_label = np.sum(weekly_label, axis = 0)

         for i in range(len(weekly_label)):
            if weekly_label[i] > 0 or weekly_label[i] == 0:
                weekly_label[i] = 1
            else:
                weekly_label[i] = 0

         return data, weekly_label,label
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    