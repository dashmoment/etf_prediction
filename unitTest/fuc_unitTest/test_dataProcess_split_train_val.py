import sys
sys.path.append('../../')
import numpy as np
from utility import data_process as dp
import hparam as conf


#=======Genererate Dummie Data=================

dummies = []
for i in range(10000):
    dummies.append([i,i])
dummies = np.vstack(dummies)

tv_gen = dp.train_validation_generaotr()
train, valid = tv_gen._split_train_val(dummies, 10, 5, 0.2)
train_s, valid_s = tv_gen._split_train_val_side_by_side(dummies, 10, 5, 0.2)

c = conf.config('test_onlyEnc_biderect_gru_nospecialstock_cls').config['common']


import random
test_result = []

for i in range(500):
     
    c['input_step'] = random.randint(10, 200)
      
    #========= generate_train_val_set_mStock with single stock ================
    train, validation , train_raw, validation_raw, _ = tv_gen.generate_train_val_set_mStock(
                                                            c['src_file_path'],['0050'], 
                                                            c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                            is_special_list = False, metafile = c['meta_file_path'])
    
    #========= generate_train_val_set_mStock with two stock ================
    train_2, validation_2 , train_raw_2, validation_raw_2, _ = tv_gen.generate_train_val_set_mStock(
                                                            c['src_file_path'],['0050', '0051'], 
                                                            c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                            is_special_list = False, metafile = c['meta_file_path'])
    
    
    if np.equal(train_2[0:len(train)],train).all() and np.equal(validation_2[0:len(validation)],validation).all():
        print('[Test] generate_train_val_set_mStock - single stock: Pass')
        test_result.appenimport random
test_result = []

for i in range(500):
     
    c['input_step'] = random.randint(10, 200)
      
    #========= generate_train_val_set_mStock with single stock ================
    train, validation , train_raw, validation_raw, _ = tv_gen.generate_train_val_set_mStock(
                                                            c['src_file_path'],['0050'], 
                                                            c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                            is_special_list = False, metafile = c['meta_file_path'])
    
    #========= generate_train_val_set_mStock with two stock ================
    train_2, validation_2 , train_raw_2, validation_raw_2, _ = tv_gen.generate_train_val_set_mStock(
                                                            c['src_file_path'],['0050', '0051'], 
                                                            c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                            is_special_list = False, metafile = c['meta_file_path'])
    
    
    if np.equal(train_2[0:len(train)],train).all() and np.equal(validation_2[0:len(validation)],validation).all():
        print('[Test] generate_train_val_set_mStock - single stock: Pass')
        test_result.append(True)
    
    else:
        print('[Test] generate_train_val_set_mStock - single stock: Fail')
        test_result.append(False)
    #========= generate_train_val_set_mStock with is_special_list=False ================
    train_m, validation_m , train_raw_m, validation_raw_m, _ = tv_gen.generate_train_val_set_mStock(
                                                            c['src_file_path'],c['input_stocks'], 
                                                            c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                            is_special_list = False, metafile = c['meta_file_path'])
    
    if np.equal(train_m[0:len(train)],train).all() and np.equal(validation_m[0:len(validation)],validation).all()\
        and np.equal(train_2[len(train):],train_m[len(train):len(train_2)]).all() \
        and np.equal(validation_2[len(validation):],validation_m[len(validation):len(validation_2)]).all():
        print('[Test] generate_train_val_set_mStock - multiple stock: Pass')
        test_result.append(True)
    
    else:
        print('[Test] generate_train_val_set_mStock - multiple stock: Fail')
        test_result.append(False)
    d(True)
    
    else:
        print('[Test] generate_train_val_set_mStock - single stock: Fail')
        test_result.append(False)
    #========= generate_train_val_set_mStock with is_special_list=False ================
    train_m, validation_m , train_raw_m, validation_raw_m, _ = tv_gen.generate_train_val_set_mStock(
                                                            c['src_file_path'],c['input_stocks'], 
                                                            c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                            is_special_list = False, metafile = c['meta_file_path'])
    
    if np.equal(train_m[0:len(train)],train).all() and np.equal(validation_m[0:len(validation)],validation).all()\
        and np.equal(train_2[len(train):],train_m[len(train):len(train_2)]).all() \
        and np.equal(validation_2[len(validation):],validation_m[len(validation):len(validation_2)]).all():
        print('[Test] generate_train_val_set_mStock - multiple stock: Pass')
        test_result.append(True)
    
    else:
        print('[Test] generate_train_val_set_mStock - multiple stock: Fail')
        test_result.append(False)
    
#========= generate_train_val_set_mStock with is_special_list=True ================
train_ms, validation_ms , train_raw_ms, validation_raw_ms, _ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'],c['input_stocks'], 
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        is_special_list = True, metafile = c['meta_file_path'])




