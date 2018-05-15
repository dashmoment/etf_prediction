import pickle

def read_metafile(filepath):
    
    f = open(filepath, 'rb')
    stockList = pickle.load(f)
    pickle.load(f)
    period = pickle.load(f)
    feature_names = pickle.load(f)
    close_price_mean_var = pickle.load(f)
    
    return close_price_mean_var, stockList, period, feature_names
