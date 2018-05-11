import sys
sys.path.append('../')
import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_specialList as dp
import model_zoo as mz
import loss_func as l
import numpy as np


tf.reset_default_graph()  
c = conf.config('test_onlyEnc_biderect_gru_allstock').config['common']
train, validation = dp.read_special_data()
sample_window = c['input_step'] + c['predict_step']


if c['feature_size'] == None: c['feature_size'] = train.shape[-1]
#x = tf.placeholder(tf.float32, [None, c['input_step'], train.shape[-1]])
x = tf.placeholder(tf.float32, [None, c['input_step'], c['feature_size']])
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 

decoder_output = mz.model_zoo(c, x, y, dropout = 0.6, is_train = True).decoder_output
decoder_output_eval = mz.model_zoo(c, x, y, dropout = 1.0, is_train = False).decoder_output

l2_reg_loss = 0
for tf_var in tf.trainable_variables():
                #print(tf_var.name)
    if not ("bias" in tf_var.name or "output_project" in tf_var.name):
        l2_reg_loss +=  tf.reduce_mean(tf.nn.l2_loss(tf_var))
        

loss = l.l1loss(decoder_output, y)
loss_eval = l.l1loss(decoder_output_eval, y)
#train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
train_op = tf.train.RMSPropOptimizer(1e-4, 0.9).minimize(loss)

with tf.name_scope('train_summary'):
    tf.summary.scalar('l2loss', loss, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('l2loss', loss_eval, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 

sess = sesswrapper.sessionWrapper(	c, x, y, loss_eval, c['input_step'],
									loss, train_op, merged_summary_train, 
									merged_summary_val)

sess.run(train, validation)




