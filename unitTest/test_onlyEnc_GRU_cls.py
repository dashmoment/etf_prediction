import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
import hparam as conf
import sessionWrapper as sesswrapper
from utility import dataProcess as dp
import model_zoo as mz
import loss_func as l

tf.reset_default_graph()  

c = conf.config('test_onlyEnc_biderect_gru_cls').config['common']
#c['src_file_path'] = '../Data/all_feature_data.pkl'

tv_gen = dp.train_validation_generaotr()
if 'random' in c['sample_type']:  tv_gen.generate_train_val_set =  tv_gen.generate_train_val_set_random
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])
sample_window = c['input_step'] + c['predict_step']
if len(np.shape(train)) > 3:
    train = np.reshape(np.transpose(train,(0,3,1,2)), (-1,sample_window,np.shape(train)[-2]))
    validation = np.reshape(np.transpose(validation,(0,3,1,2)), (-1,sample_window,np.shape(validation)[-2]))


if c['feature_size'] == None: c['feature_size'] = train.shape[-1]
#x = tf.placeholder(tf.float32, [None, c['input_step'], train.shape[-1]])
x = tf.placeholder(tf.float32, [None, c['input_step'], c['feature_size']])
y = tf.placeholder(tf.float32, [None, c['predict_step'], 3]) 

decoder_output = mz.model_zoo(c, x, y, dropout = 0.6, is_train = True).decoder_output
decoder_output_eval = mz.model_zoo(c, x, y, dropout = 1.0, is_train = False).decoder_output

predict_train = tf.argmax(tf.nn.softmax(decoder_output), axis=-1)
predict_eval = tf.argmax(tf.nn.softmax(decoder_output_eval), axis=-1)
ground_truth = tf.argmax(y, axis=-1)
accuracy_train = tf.reduce_sum(tf.cast(tf.equal(predict_train, ground_truth), tf.float32))
accuracy_eval = tf.reduce_sum(tf.cast(tf.equal(predict_eval, ground_truth), tf.float32))

l2_reg_loss = 0
for tf_var in tf.trainable_variables():
                #print(tf_var.name)
    if not ("bias" in tf_var.name or "output_project" in tf_var.name):
        l2_reg_loss +=  tf.reduce_mean(tf.nn.l2_loss(tf_var))
        

loss = l.cross_entropy_loss(decoder_output, y) 
loss_eval = l.cross_entropy_loss(decoder_output_eval, y)
#train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
train_op = tf.train.RMSPropOptimizer(1e-3, 0.9).minimize(loss)

with tf.name_scope('train_summary'):
    tf.summary.scalar('cross_entropy_loss', loss, collections=['train'])
    tf.summary.scalar('accuracy', accuracy_train, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('cross_entropy_loss', loss_eval, collections=['validatin'])
    tf.summary.scalar('accuracy', accuracy_eval, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 
    
    
sess = sesswrapper.sessionWrapper(	c, x, y, loss_eval, c['input_step'],
									loss, train_op, merged_summary_train, 
									merged_summary_val)

sess.run(train, validation)
    
#import random
#import numpy as np
#def get_batch_random_cls(data_set, train_step,batch_size, cur_index, feature_size=None):
#    
#    #data_set: [None, time_step, features ]
#    #batch_idx: index of batch start point
#
#    sample_step = train_step + 5
#
#    batch = []
#
#    for i in range(batch_size):
#        
#        rnd = random.randint(0,len(data_set)-sample_step)
#        tmpbatch =  np.reshape(data_set[rnd:rnd + sample_step, :], (1, sample_step, -1))
#        batch.append(tmpbatch)
#    
#    batch = np.squeeze(np.array(batch))
#    train, label = np.split(batch, [train_step], axis=1)
#   
#    if feature_size == None: feature_size = np.shape(train)[-1]
#    #train = np.reshape(train[:,:,3], (batch_size, train_step, -1))
#    train = train[:,:,:feature_size]
#    label = label[:,:,44:]
#
#    return train, label
#
#lloss = []
#train_val = []
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    
#    for i in range(100000):
#        
#        train_data, train_label = get_batch_random_cls(train, 100, c['batch_size'], 0,  c['feature_size'])
#        _, l, p = sess.run([train_op, loss, decoder_output], feed_dict={x:train_data, y:train_label})
#        tvars = tf.trainable_variables()
#        tvars_vals = sess.run(tvars[-2])
#        train_val.append(tvars_vals)
#        print(i, '  ', l)
#        lloss.append(l)
#    
#import matplotlib.pyplot as plt
#plt.plot(lloss)






