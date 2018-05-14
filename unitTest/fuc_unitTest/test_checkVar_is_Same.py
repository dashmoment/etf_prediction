import init
import hparam as conf
import tensorflow as tf
import data_process as dp
import os
import model_zoo as mz

c = conf.config('baseline').config['common']
c['test_period'] =  ['20130102', '20130402']

tv_gen = dp.train_validation_generaotr()
evalSet, _  = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], 0.0, c['test_period'])

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
predicted = mz.model_zoo(c, x,y, True).decoder_output

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

        
    tvars = tf.global_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        print(var.name)
       
        
    vars_op = [var for var in tf.global_variables() if var.name in ['baseline/encoder/w_encoder:0', 
                                                               'baseline/decoder/attention_wrapper/basic_lstm_cell/kernel:0',
                                                               'baseline/encoder/lstm_encoder/basic_lstm_cell/kernel:0',
                                                               'baseline/decoder/attention_wrapper/basic_lstm_cell/kernel:0',
                                                               'baseline/decoder/attention_wrapper/attention_layer/kernel:0',
                                                               'baseline/decoder/output_project/kernel:0']]
    
    w_train = sess.run(vars_op)
               


predicted = mz.model_zoo(c, x,y, False).decoder_output
    
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
            
    if load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

       
        
    vars_op_eval = [var for var in tf.global_variables() if var.name in ['baseline/encoder/w_encoder:0', 
                                                               'baseline/decoder/attention_wrapper/basic_lstm_cell/kernel:0',
                                                               'baseline/encoder/lstm_encoder/basic_lstm_cell/kernel:0',
                                                               'baseline/decoder/attention_wrapper/basic_lstm_cell/kernel:0',
                                                               'baseline/decoder/attention_wrapper/attention_layer/kernel:0',
                                                               'baseline/decoder/output_project/kernel:0']]
    
    w_eval = sess.run(vars_op_eval)
                
   
                
                
                
            