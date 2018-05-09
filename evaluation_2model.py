import hparam as conf
import tensorflow as tf
import data_process_list as dp
import os
import model_zoo as mz
import numpy as np


import pickle
f = open('/home/ubuntu/dataset/etf_prediction/all_meta_data.pkl', 'rb')
_ = pickle.load(f)
_ = pickle.load(f)
_ = pickle.load(f)
_ = pickle.load(f)
close_price_mean_var = pickle.load(f)

mean = close_price_mean_var.mean_[0]
std = np.sqrt(close_price_mean_var.var_[0])

c = conf.config('test_onlyEnc_biderect_gru').config['common']
c['checkpoint_dir'] = './model/test_onlyEnc_biderect_gru_orth_init'
sample_step = c['input_step'] +  c['predict_step']

tv_gen = dp.train_validation_generaotr()
if 'random' in c['sample_type'] :  tv_gen.generate_train_val_set =  tv_gen.generate_train_val_set_random
_ , evalSet  = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], 0.2, None)

if c['feature_size'] == None: c['feature_size'] = evalSet.shape[-1]
c['batch_size'] = len(evalSet) - sample_step

eval_data = []
for i in range(len(evalSet)-sample_step):
    eval_data.append(evalSet[i:i+sample_step])    

eval_data = np.stack(eval_data)

train, label = np.split(eval_data, [c['input_step']], axis=1)    
train = np.reshape(train[:,:,:c['feature_size']], (c['batch_size'], c['input_step'], -1))
label_reg = label[:,:,3]
label_cls = label[:,:,44:]


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
        
tf.reset_default_graph()  
x = tf.placeholder(tf.float32, [None, c['input_step'],c['feature_size']]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 
predict_reg = mz.model_zoo(c, x, y, dropout = 1.0, is_train=False).decoder_output

predict_reg_resore = predict_reg*std + + mean
label_reg_restore = label_reg*std + mean

abs_loss = tf.abs(predict_reg_resore-label_reg_restore)
weighted_array = [0.1,0.15,0.2,0.25,0.3]
score_ref = ((label_reg_restore -abs_loss)/label_reg_restore)*0.5
w_score = weighted_array*score_ref
mean_score = tf.reduce_mean(tf.reduce_sum(w_score, axis=1))


with tf.Session() as sess:
    
    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        
   
    predict = sess.run(predict_reg_resore, feed_dict={x:train, y:label_reg})
    loss, plain_scores, w_scores , mean_score_reg = sess.run([abs_loss,  score_ref, w_score, mean_score], feed_dict={x:train, y:label_reg})
    
    sess.close()
                
tf.reset_default_graph()              
c = conf.config('test_onlyEnc_biderect_gru_cls').config['common']
c['checkpoint_dir'] = './model/test_onlyEnc_biderect_gru_orth_init_cls'
sample_step = c['input_step'] +  c['predict_step']


if c['feature_size'] == None: c['feature_size'] = evalSet.shape[-1]
c['batch_size'] = len(evalSet) - sample_step


x = tf.placeholder(tf.float32, [None, c['input_step'],c['feature_size']]) 
y = tf.placeholder(tf.float32, [None, c['predict_step'], 3]) 

logits = mz.model_zoo(c, x, y, dropout = 1.0, is_train = False).decoder_output
softmax_out = tf.nn.softmax(logits)
predict_cls = tf.argmax(tf.nn.softmax(logits), axis=-1)
ground_truth = tf.argmax(y, axis=-1)

weighted_array = [0.1,0.15,0.2,0.25,0.3]
plane_result = tf.cast(tf.equal(predict_cls, ground_truth), tf.float32)
weighted_result = plane_result*0.5*weighted_array
score_tf = tf.reduce_sum(weighted_result, axis=1)
mean_score = tf.reduce_mean(score_tf)

with tf.Session() as sess:
    
    
    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        
    l, s, p = sess.run([logits, softmax_out,predict_cls], feed_dict={x:train, y:label_cls})
    result_cls, score, w_score, mean_score_cls = sess.run([plane_result, score_tf, weighted_result, mean_score], feed_dict={x:train, y:label_cls})
    

print("Total Score:{}, Reg Score:{}, Cls Score:{}".format(mean_score_reg+mean_score_cls, mean_score_reg, mean_score_cls))
   