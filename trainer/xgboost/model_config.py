import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

    
def model_config():
        conf = {  
        
                'xgb':{
                            'model': xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=False, 
                                              objective='multi:softmax', num_class=3),
                            'param': { 
                                          'learning_rate': np.arange(0.01, 0.2, 0.05),
                                          'max_depth': np.arange(3, 10),
                                          #'subsample':np.arange(0.5, 1, 0.2),
                                          'min_child_weight': np.arange(1,6,2),
                                          #'n_estimators': np.arange(100,600,100)
                                     },
                             'fit_param':True                        
                      },
        
                'rf': {
                            'model': RandomForestClassifier(n_estimators = 100),
                            'param': { 
                                          #'min_samples_split' : [2, 5, 10],
                                          #'min_samples_leaf' : [1, 2, 4],
                                          'max_depth' : np.arange(10,100,20),
                                          'max_features' : ['auto', 'sqrt', 'log2'],
                                          'n_estimators': np.arange(500,1000,200),
                                          #'bootstrap' : [True, False]
                                     },
                            'fit_param':True                              
                        },
        
                'svc':{
                            'model': svm.SVC(),
                            'param': {
                                        'kernel':[ 'linear']
                                     },
                            
                            'fit_param':False                                           
                        }     
                 }
                            
        return conf