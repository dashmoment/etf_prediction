import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_list as dp
import model_zoo as mz
import numpy as np

def l2loss(x,y):   
    loss = tf.reduce_mean(tf.squared_difference(x, y))
    return loss 

def cross_entropy_loss(x,y):   
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))
    return loss 

c = conf.config('baseline_2in1').config['common']

tv_gen = dp.train_validation_generaotr()
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])


x = tf.placeholder(tf.float32, [None, c['input_step'], np.shape(train)[-1]]) 
y = tf.placeholder(tf.float32, [None, c['predict_step'], 4]) 


decoder_output = mz.model_zoo(c, x,y, dropout = 0.6, is_train = True).decoder_output
decoder_output_eval = mz.model_zoo(c, x,y, dropout = 1.0, is_train = False).decoder_output

decoder_output_price = tf.slice(decoder_output, [0,0,0], [c['batch_size'], c['predict_step'], 1])
decoder_output_ud= tf.slice(decoder_output, [0,0,1], [c['batch_size'], c['predict_step'], 3])

decoder_output_eval_price = tf.slice(decoder_output_eval, [0,0,0], [c['batch_size'], c['predict_step'], 1])
decoder_output_eval_ud = tf.slice(decoder_output_eval, [0,0,1], [c['batch_size'], c['predict_step'], 3])

y_price = tf.slice(y, [0,0,0], [c['batch_size'], c['predict_step'], 1])
y_ud = tf.slice(y, [0,0,1], [c['batch_size'], c['predict_step'], 3])

predict_train = tf.argmax(decoder_output, axis=-1)
predict_eval = tf.argmax(decoder_output_eval, axis=-1)
ground_truth = tf.argmax(y_ud, axis=-1)

accuracy_train = tf.reduce_sum(tf.cast(tf.equal(predict_train, ground_truth), tf.float32))
accuracy_eval = tf.reduce_sum(tf.cast(tf.equal(predict_eval, ground_truth), tf.float32))

l2loss_train = l2loss(decoder_output_price, y_price)
loss_ud = cross_entropy_loss(decoder_output_ud, y_ud)
l2loss_eval = l2loss(decoder_output_eval_price, y_price)
loss_eval_ud = cross_entropy_loss(decoder_output_eval_ud, y_ud)

loss = l2loss_train + loss_ud
loss_eval = l2loss_eval +  loss_eval_ud
train_op = tf.train.RMSPropOptimizer(1e-4, 0.9).minimize(loss)

with tf.name_scope('train_summary'):
    tf.summary.scalar('l2loss', loss, collections=['train'])
    tf.summary.scalar('accuracy', accuracy_train, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('l2loss', loss_eval, collections=['validatin'])
    tf.summary.scalar('accuracy', accuracy_eval, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 
    
with tf.Session() as sess:
    
#    sess.run(tf.global_variables_initializer())
#    predict = sess.run([decoder_output_eval_price, decoder_output_eval_ud], feed_dict={x:train[0:8, 0:10, 0:4], y:train[0:8, 10:15, 3:7]})


    sess = sesswrapper.sessionWrapper(	c, x, y, loss_eval, c['input_step'],
    									loss, train_op, merged_summary_train, 
    									merged_summary_val)
    sess.run(train, validation)




