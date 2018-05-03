import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_list as dp
import model_zoo as mz


def l2loss(x,y):   
    loss = tf.reduce_mean(tf.squared_difference(x, y))
    return loss 

c = conf.config('sample').config['common']

tv_gen = dp.train_validation_generaotr()
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])

x = tf.placeholder(tf.float32, [None, c['input_step'], train.shape[-1]]) 
y = tf.placeholder(tf.float32, [None, c['predict_step']]) 

decoder_output = mz.model_zoo(c, x,y, dropout = 0.6, is_train = True).decoder_output
decoder_output_eval = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output

loss = l2loss(decoder_output, y)
loss_eval = l2loss(decoder_output_eval, y)
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




