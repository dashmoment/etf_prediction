import pandas as pd
import numpy as np
import hparam as conf
import data_process_list as dp
import tensorflow as tf
import os
import model_zoo as mz

def map_ud(softmax_output):
    
    ud_meta = {0:-1, 1:0, 2:1}
    ud_index = np.argmax(softmax_output, axis=-1)
    ud = [ud_meta[v] for v in ud_index]
    
    return ud



c = conf.config('baseline_2in1').config['common']    

stocks = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208', '00690', '00692', '00701', '00713']

tv_gen = dp.train_validation_generaotr()  
eval_set  = tv_gen.generate_test_set(c['src_file_path'], stocks, c['input_step'])

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

c['batch_size'] = 1
n_linear_hidden_units = c['n_linear_hidden_units']
n_lstm_hidden_units =  c['n_lstm_hidden_units']
x = tf.placeholder(tf.float32, [None, c['input_step'], 13]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 

logits = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output
predicted_price = tf.slice(logits, [0,0,0], [c['batch_size'], c['predict_step'], 1])
predicted_ud= tf.nn.softmax(tf.slice(logits, [0,0,1], [c['batch_size'], c['predict_step'], 3]))

predict = {}

with tf.Session() as sess:

    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        
        
    for s in eval_set:
        train = np.expand_dims(eval_set[s], axis=0)
        price, ud = sess.run([predicted_price, predicted_ud] , feed_dict={x:train})
        price = price[0,:,0]
        ud = ud[0]
        ud = map_ud(ud)
        
        predict[s] = {'price':price, 'ud':ud}
        

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