import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

  

class model_config:

  def __init__(self, conf_name):

    self.get = self.get_config(conf_name)

  def get_config(self, conf_name):

        try:
            conf = getattr(self, conf_name)
            return conf()

        except: 
            print("Can not find configuration")
            raise

  def xgb(self):

      conf = {
                'model': xgb.XGBClassifier( max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True, 
                                            objective='multi:softmax', num_class=3
                                           ),
                'param': {
                              'max_depth': np.arange(3, 4, 3),
                              #'min_child_weight': np.arange(1,2,3)
                              'learning_rate': np.arange(0.01, 0.15, 0.05),
                              'max_depth': np.arange(3, 10, 3),
                              #'subsample':np.arange(0.5, 1, 0.2),
                              'min_child_weight': np.arange(1,6,3),
                              #'n_estimators': np.arange(400,600,100)
                         },
                 'fit_param':True                        
              }

      return conf

  def xgb_2cls(self):

      conf = {
                'model': xgb.XGBClassifier( max_depth=3, learning_rate=0.05 ,n_estimators=500, silent=True, 
                                            objective='multi:softmax', num_class=2
                                           ),
                'param': {
                              'max_depth': np.arange(3, 4, 3),
                              #'min_child_weight': np.arange(1,2,3)
                              'learning_rate': np.arange(0.01, 0.15, 0.05),
                              'max_depth': np.arange(3, 10, 3),
                              #'subsample':np.arange(0.5, 1, 0.2),
                              'min_child_weight': np.arange(1,6,3),
                              'n_estimators': np.arange(400,600,100)
                         },
                 'fit_param':True                        
              }

      return conf

  def rf(self):
     
      conf = {
                  'model': RandomForestClassifier(n_estimators = 500),
                  'param': { 
                                #'min_samples_split' : [2, 5, 10],
                                'min_samples_leaf' : [1, 2, 4],
                                'max_depth' : np.arange(40,100,20),
                                #'max_features' : ['auto', 'sqrt', 'log2'],
                                #'n_estimators': np.arange(500,1000,200),
                                #'bootstrap' : [True, False]
                           },
                  'fit_param':True                      
              }

      return conf

  def svc(self):
     
      conf = {
                  'model': svm.SVC(),
                  'param': {
                                'C':[0.001, 0.01, 0.1, 1, 10],
                                'gamma':[0.001, 0.01, 0.1, 1],
                                'kernel':[ 'linear', 'rbf']
                             },
                    
                  'fit_param':False                
              }

      return conf


  def stack(self):

      conf = {
               
                'param': {},                 
                'fit_param':True                        
              }

      return conf