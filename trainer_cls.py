import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process as dp
import model_zoo as mz


def l2loss(x,y):   
    loss = tf.reduce_mean(tf.squared_difference(x, y))
    return loss 

def cross_entropy_loss(x,y):   
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))
    return loss 

c = conf.config('baseline_cls').config['common']

tv_gen = dp.train_validation_generaotr()
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])

import numpy as np
train_dummies_label = np.ones((train.shape[0],train.shape[1],3))*0.3
eval_dummies_label = np.ones((validation.shape[0], validation.shape[1], 3))*0.3

train = np.dstack([train, train_dummies_label])
validation = np.dstack([validation, eval_dummies_label])

features = 4
labels = 3

x = tf.placeholder(tf.float32, [None, c['input_step'], features]) 
y = tf.placeholder(tf.float32, [None, c['predict_step'], labels]) 

decoder_output = mz.model_zoo(c, x,y, dropout = 0.6, is_train = True).decoder_output
decoder_output_eval = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output

predict = tf.argmax(decoder_output_eval, axis=-1)
ground_truth = tf.argmax(y, axis=-1)

accuracy = tf.metrics.accuracy(
    ground_truth,
    predict,
)


loss = cross_entropy_loss(decoder_output, y)
loss_eval = cross_entropy_loss(decoder_output_eval, y)
train_op = tf.train.RMSPropOptimizer(1e-4, 0.9).minimize(loss)

with tf.name_scope('train_summary'):
    tf.summary.scalar('l2loss', loss, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('l2loss', loss_eval, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 
    
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    predict = sess.run([ground_truth, accuracy], feed_dict={x:train[0:8, 0:10, 0:4], y:train[0:8, 10:15, 4:7]})
    

#sess = sesswrapper.sessionWrapper(	c, x, y, loss_eval, c['input_step'],
#									loss, train_op, merged_summary_train, 
#									merged_summary_val)
#
#sess.run(train, validation)




