import hparam as conf
import tensorflow as tf
import data_process as dp
import os
import model_zoo as mz
import numpy as np

c = conf.config('baseline').config['common']

tv_gen = dp.train_validation_generaotr()
evalSet, _  = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], 0.0, c['eval_period'])

c['batch_size'] = len(evalSet)

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
        

batch_size = evalSet.shape[0]
n_linear_hidden_units = c['n_linear_hidden_units']
n_lstm_hidden_units =  c['n_lstm_hidden_units']


x = tf.placeholder(tf.float32, [None, c['input_step'], evalSet.shape[-1]]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 
isTrain = tf.placeholder(tf.bool, ())  
predicted = mz.model_zoo(c, x, y, False).decoder_output

abs_loss = tf.abs(predicted-y)
weighted_array = [0.1,0.15,0.2,.025,0.3]
score = ((y -abs_loss)/y)*0.5
w_score = weighted_array*score
mean_score = tf.reduce_mean(tf.reduce_sum(w_score, axis=1))

with tf.Session() as sess:
    
    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        
   
    train, label = np.split(evalSet, [c['input_step']], axis=1)    
    predict = sess.run(predicted, feed_dict={x:train, y:label[:,:,-1], isTrain:False})
    loss, plain_scores, w_scores ,mean_w_scores = sess.run([abs_loss,  score, w_score, mean_score], feed_dict={x:train, y:label[:,:,-1], isTrain:False})
                
                
                
            