import sys
sys.path.append('../')
import tensorflow as tf
import hparam as conf
import sessionWrapper as sesswrapper
import data_process_list as dp
import model_zoo as mz
import loss_func as l

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
train = []
validation = [] 

tf.reset_default_graph()  

c = conf.config('test_onlyEnc_biderect_gru_allstock_cls').config['common']
sample_step = c['input_step'] +  c['predict_step']
rand = 0
for i in range(0, 38500, 35):
    
    rand = random.random() * 2 * math.pi
    
#    if rand >= 1:
#        rand = 0
#    else:
#        rand += 0.1
#    
#    rand = rand * 2 * math.pi
    
    if i < 35000:

        temp_sig = np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, sample_step)
        train.append([temp_sig, np.sin(temp_sig)])        
    else:

        temp_sig = np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, sample_step)
        validation.append([temp_sig, np.sin(temp_sig)])                

train = np.transpose(np.array(train), (0,2,1))
validation = np.transpose(np.array(validation), (0,2,1))

#Check method is correct
plt.scatter(train[0,:,0], train[0,:,1])
plt.scatter(train[1,:,0], train[1,:,1])

def gen_label(pre, cur):
    
    if cur - pre > 0:
        return [0,0,1]
    elif cur-pre < 0:
        return [1,0,0]
    else:
        return [0,1,0]

def gen_labelSet(dataSet):
    train_labels = []
    for i in range(len(dataSet)):
        tmp_lable = []
        for j in range(c['input_step'],sample_step):
            tmp_lable.append(gen_label(dataSet[i,j-1,1], dataSet[i,j,1]))
            
        train_labels.append(tmp_lable)
    return train_labels

train_data= train[:,:c['input_step'],:]
train_labels = np.array(gen_labelSet(train))
validation_data = validation[:,:c['input_step'],:]
validation_labels = np.array(gen_labelSet(validation))

x = tf.placeholder(tf.float32, [None, c['input_step'], 2])
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
train_op = tf.train.RMSPropOptimizer(1e-4, 0.9).minimize(loss)

    
    
#plt.scatter(validation[0:55,:,0], validation[0:55,:,1])

batchsize = 32
data_log = []
loss_log = []
loss_eval_log = []

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    for i in tqdm(range(2000)):
    
        data = list(zip(train_data, train_labels))
        np.random.shuffle(data)
        train_batch, label_batch = zip(*data)
    
        data_log.append(train_batch[0:batchsize])
    
        l, _ = sess.run([loss, train_op], feed_dict={x:train_batch[0:batchsize], y:label_batch[0:batchsize]})
        l_eval = sess.run(loss_eval, feed_dict={x:validation_data[0:batchsize], y:validation_labels[0:batchsize]})
        loss_log.append(l)
        loss_eval_log.append(l_eval)
        
    test_p, test_acc = sess.run([predict_eval, accuracy_eval], feed_dict={x:validation_data[0:batchsize], y:validation_labels[0:batchsize]})
    

test_truth = np.argmax(validation_labels[0:batchsize], axis = -1)
plt.figure()
plt.plot(validation[0,50:,1], color='b')
plt.plot(np.argmax(validation_labels[0], axis = -1), color='g')
plt.plot(test_p[0], color='r')
plt.figure()
plt.plot(validation[1,50:,1], color='b')
plt.plot(np.argmax(validation_labels[1], axis = -1), color='g')
plt.plot(test_p[1], color='r')
plt.figure()
plt.plot(validation[2,50:,1], color='b')
plt.plot(np.argmax(validation_labels[2], axis = -1), color='g')
plt.plot(test_p[2], color='r')

plt.figure()
plt.plot(loss_log)
plt.plot(loss_eval_log)
