import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process as dp
import model_zoo as mz
import netFactory as nf
import numpy as np

c = conf.config('sample').config['common']
filepath = './Data/all_data.pkl'


tv_gen = dp.train_validation_generaotr()
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], ['1101'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])
#train, validation = tv_gen.generate_train_val_set(filepath, ['0050'], train_step, predict_step, 0.15, None)

x = tf.placeholder(tf.float32, [None, c['input_step'], train.shape[-1]]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 
isTrain = tf.placeholder(tf.bool, ())  

decoder_output = mz.baseline_LuongAtt_lstm(	x, c['batch_size'], y, c['n_linear_hidden_units'], c['n_lstm_hidden_units'],
											c['n_attlstm_hidden_units'], c['n_att_hidden_units'], 
											isTrain, reuse = False)

loss = tf.losses.mean_squared_error(y, decoder_output)
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


with tf.name_scope('train_summary'):
    tf.summary.scalar('l2loss', loss, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('l2loss', loss, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 

sess = sesswrapper.sessionWrapper(	x, y, isTrain, c['input_step'],
									loss, train_op, merged_summary_train, 
									merged_summary_val, './model/test', 'test.ckpt')

sess.run(c['batch_size'], train, validation)




