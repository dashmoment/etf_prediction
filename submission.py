import pandas as pd
import numpy as np
import hparam as conf
import data_process_list as dp
import tensorflow as tf
import os
import model_zoo as mz

import pickle
f = open('/home/ubuntu/dataset/etf_prediction/all_meta_data_Nm[0]_59.pkl', 'rb')
_ = pickle.load(f)
_ = pickle.load(f)
_ = pickle.load(f)
l = pickle.load(f)
close_price_mean_var = pickle.load(f)

def map_ud(softmax_output):
    
    ud_meta = {0:-1, 1:0, 2:1}
    ud_index = np.argmax(softmax_output, axis=-1)
    ud = [ud_meta[v] for v in ud_index]
    
    return ud

def load_ckpt(saver, sess, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")

        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt  and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False 




stocks = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208']
#'00690', '00692', '00701', '00713'


############### Predict Price ###############
c = conf.config('test_onlyEnc_biderect_gru_mstock').config['common']   
c['batch_size'] = 1
tv_gen = dp.train_validation_generaotr()  
eval_set  = tv_gen.generate_test_set(c['src_file_path'], stocks, c['input_step']) 

tf.reset_default_graph() 
x = tf.placeholder(tf.float32, [None, c['input_step'], c['feature_size']]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 

predict_price = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output

predict = {}

with tf.Session() as sess:

    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        
        
    for s in eval_set:
        train = np.expand_dims(eval_set[s], axis=0)
        train = train[:,:,:c['feature_size']]
        price = sess.run(predict_price , feed_dict={x:train})
        price = price[0,:]
        predict[s] = {'price':price*np.sqrt(close_price_mean_var.var_) + close_price_mean_var.mean_ }


############### Predict up-down ###############

c = conf.config('test_onlyEnc_biderect_gru_cls').config['common']   
c['batch_size'] = 1 
tv_gen = dp.train_validation_generaotr()  
eval_set  = tv_gen.generate_test_set(c['src_file_path'], stocks, c['input_step']) 

tf.reset_default_graph() 

x = tf.placeholder(tf.float32, [None, c['input_step'], c['feature_size']]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 

logits = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output
predicted_ud= tf.nn.softmax(logits)


with tf.Session() as sess:

    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        
        
    for s in eval_set:
        train = np.expand_dims(eval_set[s], axis=0)
        train = train[:,:,:c['feature_size']]
        ud = sess.run(predicted_ud , feed_dict={x:train})
        ud_mapped = map_ud(ud[0])
        predict[s]['ud'] = ud_mapped
        

columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
          'Wed_ud', 'Wed_cprice	', 'Thu_ud', 'Thu_cprice', 'Fri_ud',	'Fri_cprice']

df = pd.DataFrame(columns=columns)  
idx = 0

for s in stocks:     
    results = [s]
    for i in range(5):
        results.append(predict[s]['price'][i])
        results.append(predict[s]['ud'][i])
    
    df.loc[idx] = results
    idx+=1

df = df.set_index('ETFid') 
df.to_csv('./submission/sample_submit.csv', sep=',')

#import pickle
#s = open('/home/ubuntu/dataset/etf_prediction/all_mata_data.pkl', 'rb')
#a = pickle.load(s)
#b = pickle.load(s)
#c = pickle.load(s)
#fl =  pickle.load(s)  
#e =  pickle.load(s) 
#f =  pickle.load(s)  
#
#price = price*e.var_ + e.mean_