import sys
sys.path.append('../../')
from utility import dataProcess as dp
import numpy as np
import hparam as conf
import sessionWrapper as sess

c = conf.config('test_onlyEnc_biderect_gru_nospecialstock_relu_cls').config['common']
sample_window = c['input_step'] + c['predict_step']


tv_gen = dp.train_validation_generaotr()
train, validation , train_raw, validation_raw, _ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'],c['input_stocks'], 
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        metafile = c['meta_file_path'])

test_result = []
for i in range(np.shape(train)[-1]):
    c['feature_mask'] =  [i]
    c['feature_size'] = len(c['feature_mask'])
    data, label = sess.get_batch_cls(train, c['input_step'],  c['batch_size'], 0, c['feature_mask'])
    
    test_sample =  np.reshape(train[0: c['batch_size'],:c['input_step'],i], (-1,c['input_step'],1))
    is_pass = np.equal(data,test_sample).all()
    
    test_result.append(is_pass)
    
if all(test_result):
    print("[Test] single feature: Pass")
else:
    print("[Test] single feature: Fail")
    
    
test_result = []
for i in range(0, np.shape(train)[-1]-1):
    c['feature_mask'] =  [i, i+1]
    c['feature_size'] = len(c['feature_mask'])
    data, label = sess.get_batch_cls(train, c['input_step'],  c['batch_size'], 0, c['feature_mask'])
    test_sample = train[0: c['batch_size'],:c['input_step'],i:i+2]   
    is_pass = np.equal(data, test_sample).all()    
    test_result.append(is_pass)
    
if all(test_result):
    print("[Test] Two feature: Pass")
else:
    print("[Test] two feature: Fail")
    
test_result = []
for i in range(0, np.shape(train)[-1]-10):
    c['feature_mask'] =  list(range(i,i+10))
    c['feature_size'] = len(c['feature_mask'])
    data, label = sess.get_batch_cls(train, c['input_step'],  c['batch_size'], 0, c['feature_mask'])
    test_sample = train[0: c['batch_size'],:c['input_step'],i:i+10]   
    is_pass = np.equal(data, test_sample).all()
    
    test_result.append(is_pass)
    
if all(test_result):
    print("[Test] Ten feature: Pass")
else:
    print("[Test] Ten feature: Fail")
    


c['feature_mask'] =  [3,10,5]
c['feature_size'] = len(c['feature_mask'])
data, label = sess.get_batch_cls(train, c['input_step'],  c['batch_size'], 0, c['feature_mask'])

test_sample = []
mask = sorted(c['feature_mask'])
for idx in mask:
    test_sample.append(train[0: c['batch_size'],:c['input_step'],idx])
    
test_sample = np.transpose(np.stack(test_sample, axis= 0), (1,2,0))
is_pass = np.equal(data, test_sample).all()

if is_pass:
    print("[Test] Arbitrary feature: Pass")
else:
    print("[Test] Arbitrary feature: Fail")
