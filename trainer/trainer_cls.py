import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_list as dp
import model_zoo as mz
import numpy as np
import loss_func as l


c = conf.config('baseline_cls').config['common']

tv_gen = dp.train_validation_generaotr()
if c['sample_type'] == 'random' :  tv_gen.generate_train_val_set =  tv_gen.generate_train_val_set_random
train, validation = tv_gen.generate_train_val_set(c['src_file_path'],c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])

if len(np.shape(train)) > 3:
    train = np.reshape(np.transpose(train,(0,3,1,2)), (-1,35,13))
    validation = np.reshape(np.transpose(validation,(0,3,1,2)), (-1,35,13))

x = tf.placeholder(tf.float32, [None, c['input_step'], np.shape(train)[-1]]) 
y = tf.placeholder(tf.float32, [None, c['predict_step'], 3]) 

decoder_output = mz.model_zoo(c, x,y, dropout = 0.4, is_train = True).decoder_output
decoder_output_eval = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output

predict_train = tf.argmax(tf.nn.softmax(decoder_output), axis=-1)
predict_eval = tf.argmax(tf.nn.softmax(decoder_output_eval), axis=-1)
ground_truth = tf.argmax(y, axis=-1)

accuracy_train = tf.reduce_sum(tf.cast(tf.equal(predict_train, ground_truth), tf.float32))
accuracy_eval = tf.reduce_sum(tf.cast(tf.equal(predict_eval, ground_truth), tf.float32))

loss = l.cross_entropy_loss(decoder_output, y)
loss_eval = l.cross_entropy_loss(decoder_output_eval, y)
train_op = tf.train.RMSPropOptimizer(1e-4, 0.9).minimize(loss)

with tf.name_scope('train_summary'):
    tf.summary.scalar('cross_entropy_loss', loss, collections=['train'])
    tf.summary.scalar('accuracy', accuracy_train, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('cross_entropy_loss', loss_eval, collections=['validatin'])
    tf.summary.scalar('accuracy', accuracy_eval, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 
    
with tf.Session() as sess:
    
#    sess.run(tf.global_variables_initializer())
#    predict = sess.run([accuracy_train, accuracy_eval, decoder_output, decoder_output_eval], feed_dict={x:train[0:16, 0:30, :], y:train[0:16, 30:, 10:13]})
#    res = sess.run(decoder_output_eval, feed_dict={x:train[0:16, 0:30, :], y:train[0:16, 30:, 10:13]})

    sess = sesswrapper.sessionWrapper(	c, x, y, accuracy_eval, c['input_step'],
    									loss, train_op, merged_summary_train, 
    									merged_summary_val)
    sess.run(train, validation)




