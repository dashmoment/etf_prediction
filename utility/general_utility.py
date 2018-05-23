import pickle

def read_metafile(filepath):
    
    f = open(filepath, 'rb')
    stockList = pickle.load(f)
    pickle.load(f)
    period = pickle.load(f)
    feature_names = pickle.load(f)
    close_price_mean_var = pickle.load(f)
    
    return close_price_mean_var, stockList, period, feature_names

def read_datefile(filepath):
    
    f = open(filepath, 'rb')
    date = pickle.load(f)
   
    return date

def map_ud(predict):

    mapper = {0:-1, 1:0, 2:1}

    return mapper[predict]