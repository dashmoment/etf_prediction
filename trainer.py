import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_list as dp
import model_zoo as mz
import numpy as np
import loss_func as l

c = conf.config('baseline').config['common']

tv_gen = dp.train_validation_generaotr()
if c['sample_type'] == 'random' :  tv_gen.generate_train_val_set =  tv_gen.generate_train_val_set_random
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])

if c['feature_size'] == None: c['feature_size'] = train.shape[-1]
if len(np.shape(train)) > 3:
    train = np.reshape(np.transpose(train,(0,3,1,2)), (-1,c['input_step'] + c['predict_step'], c['feature_size']))
    validation = np.reshape(np.transpose(validation,(0,3,1,2)), (-1,c['input_step'] + c['predict_step'],c['feature_size']))

if c['feature_size'] == None: c['feature_size'] = train.shape[-1]
#x = tf.placeholder(tf.float32, [None, c['input_step'], train.shape[-1]])
x = tf.placeholder(tf.float32, [None, c['input_step'], c['feature_size']])
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 

decoder_output = mz.model_zoo(c, x, y, dropout = 0.6, is_train = False).decoder_output
decoder_output_eval = mz.model_zoo(c, x, y, dropout = 1.0, is_train = False).decoder_output

loss = l.l2loss(decoder_output, y)
loss_eval = l.l2loss(decoder_output_eval, y)
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
#train_op = tf.train.RMSPropOptimizer(1e-4, 0.9).minimize(loss)

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




