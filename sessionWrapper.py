import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
from utility import tf_utility as ut

def gather_features(data, feature_mask):

    mask = np.zeros(np.shape(data)[-1], dtype= bool)

    for i in range(len(mask)):
        if i in feature_mask:
            mask[i] = True
        else:
            mask[i] = False

    return data[:,:,mask]


def get_batch(data_set, train_step,batch_size, cur_index, feature_mask,feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    train = gather_features(train, feature_mask)
    label = label[:,:,3]

    return train, label

def get_batch_random(data_set, train_step,batch_size, cur_index, feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    sample_step = train_step + 5

    batch = []

    for i in range(batch_size):
        
        rnd = random.randint(0,len(data_set)-sample_step)
        tmpbatch =  np.reshape(data_set[rnd:rnd + sample_step, :], (1, sample_step, -1))
        batch.append(tmpbatch)
    
    batch = np.squeeze(np.array(batch))
    train, label = np.split(batch, [train_step], axis=1)
   
    if feature_size == None: feature_size = np.shape(train)[-1]
    #train = np.reshape(train[:,:,3], (batch_size, train_step, -1))
    train = train[:,:,:feature_size]
    label = label[:,:,3]

    return train, label


def get_batch_random_cls(data_set, train_step,batch_size, cur_index, feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    sample_step = train_step + 5

    batch = []

    for i in range(batch_size):
        
        rnd = random.randint(0,len(data_set)-sample_step)
        tmpbatch =  np.reshape(data_set[rnd:rnd + sample_step, :], (1, sample_step, -1))
        batch.append(tmpbatch)
    
    batch = np.squeeze(np.array(batch))
    train, label = np.split(batch, [train_step], axis=1)
   
    if feature_size == None: feature_size = np.shape(train)[-1]
    #train = np.reshape(train[:,:,3], (batch_size, train_step, -1))
    train = train[:,:,:feature_size]
    label = label[:,:,-3:]

    return train, label


def get_batch_cls(data_set, train_step, batch_size, cur_index, feature_mask, feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    train = gather_features(train, feature_mask)
    label = label[:,:,-3:]

    return train, label

def get_batch_2in1(data_set, train_step,batch_size, cur_index,  feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    if feature_size == None: feature_size = np.shape(train)[-1]
    train = train[:,:,:feature_size]
    
    label_ud = label[:,:,-3:]
    label_p = label[:,:,3]
    label = np.dstack([label_p,label_ud])

    return train, label

def get_batch_test(data_set,train_step,batch_size, cur_index, feature_size=None):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    train = train[:,:,0]
    label = label[:,:,1]
    
    return np.reshape(train, [-1,30,1]), label

class sessionWrapper:

    def __init__(	self, conf, input_p:tf.placeholder, label_p:tf.placeholder, loss_eval,
					train_step, loss, train_op, train_summary, test_summary):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(os.path.join(conf['checkpoint_dir'], 'log'), self.sess.graph)
        
        self.x = input_p #tf.placeholder(tf.float32, [None, train_step, train_set.shape[-1]]) 
        self.y = label_p #tf.placeholder(tf.float32, [None, test_step])  
        self.loss_eval = loss_eval
        self.saver = tf.train.Saver()
        self.loss = loss
        self.optimizer = train_op
        self.train_step = train_step
        self.train_summary = train_summary
        self.test_summary = test_summary
        self.conf = conf
        self.sample_method = {'reg':get_batch,
                              'cls':get_batch_cls,
                              '2in1':get_batch_2in1,
                              'random':get_batch_random,
                              'random_cls':get_batch_random_cls,
                              'test':get_batch_test
                              }
                              

    def run(self, train_set, test_set):
        
        get_batch = self.sample_method [self.conf['sample_type']]
        print('Sample Type:{}'.format(self.conf['sample_type']))
        
        epoch =  self.conf['current_epoch']
        pbar = tqdm(total =  self.conf['total_epoch'])
        pbar.update(epoch)
        eval_bar = tqdm(total =  self.conf['total_epoch'])
        eval_bar.update(epoch)
        
        batch_size =  self.conf['batch_size']
        Nbatch = len(train_set)//batch_size
        if Nbatch > 20:
            Nbatch = 20
        else:
            Nbatch = Nbatch
            
        with self.sess as sess:
            
            if ut.load_ckpt(self.saver, sess, self.conf['checkpoint_dir'], self.conf['ckpt_name']):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            while epoch <  self.conf['total_epoch']:

                epoch += 1
                pbar.update(1)
                if  'random' not in self.conf['sample_type']: 
                    pbar.set_description('Shuffle Data')
                    np.random.shuffle(train_set)

                for i in range(Nbatch):
                    batch_index = i*batch_size
                    train_data, train_label = get_batch(train_set, self.train_step, batch_size, batch_index,  self.conf['feature_mask'], self.conf['feature_size'])
                    sess.run(self.optimizer, feed_dict={self.x:train_data, self.y:train_label})

                if epoch% self.conf['save_ckpt_epoch'] == 0:
                    train_data, train_label = get_batch(train_set, self.train_step, batch_size, random.randint(0,len(train_set)//batch_size), self.conf['feature_mask'],self.conf['feature_size'])
                    l2loss, train_sum =  sess.run([self.loss, self.train_summary], feed_dict={self.x:train_data, self.y:train_label})
                    self.summary_writer.add_summary(train_sum, epoch)
                    ut.save_ckpt(self.saver, sess, self.conf['checkpoint_dir'], self.conf['ckpt_name'], epoch)
                    pbar.set_description('train l2loss: {}'.format(l2loss))    
				
                if epoch% self.conf['evaluation_epoch']  == 0:
                    val_data, val_label = get_batch(test_set, self.train_step, batch_size, random.randint(0,len(test_set)//batch_size), self.conf['feature_mask'],self.conf['feature_size'])
                    l2loss, val_sum =  sess.run([self.loss_eval, self.test_summary], feed_dict={self.x: val_data, self.y: val_label})         
                    self.summary_writer.add_summary(val_sum, epoch)
                    eval_bar.update(self.conf['evaluation_epoch'])
                    eval_bar.set_description('Eval l2loss: {}'.format(l2loss))
                    
                    


#class sessionWrapper_cls:
#
#    def __init__(	self, conf, input_p:tf.placeholder, label_p:tf.placeholder, loss_eval,
#					train_step, loss, train_op, train_summary, test_summary):
#        self.sess = tf.Session()
#        self.sess.run(tf.global_variables_initializer())
#        self.summary_writer = tf.summary.FileWriter(os.path.join(conf['checkpoint_dir'], 'log'), self.sess.graph)
#        
#        self.x = input_p #tf.placeholder(tf.float32, [None, train_step, train_set.shape[-1]]) 
#        self.y = label_p #tf.placeholder(tf.float32, [None, test_step])  
#        self.loss_eval = loss_eval
#        self.saver = tf.train.Saver()
#        self.loss = loss
#        self.optimizer = train_op
#        self.train_step = train_step
#        self.train_summary = train_summary
#        self.test_summary = test_summary
#        self.conf = conf
#
#    def run(self, train_set, test_set):
#        
#        epoch =  self.conf['current_epoch']
#        pbar = tqdm(total =  self.conf['total_epoch'])
#        pbar.update(epoch)
#        eval_bar = tqdm(total =  self.conf['total_epoch'])
#        eval_bar.update(epoch)
#        
#        batch_size =  self.conf['batch_size']
#        Nbatch = len(train_set)//batch_size
#        
#        with self.sess as sess:
#            
#            if load_ckpt(self.saver, sess, self.conf['checkpoint_dir'], self.conf['ckpt_name']):
#                print(" [*] Load SUCCESS")
#            else:
#                print(" [!] Load failed...")
#
#            while epoch <  self.conf['total_epoch']:
#                epoch += 1
#                pbar.update(1)
#                np.random.shuffle(train_set)
#
#                for i in range(Nbatch):
#                    batch_index = i*batch_size
#                    train_data, train_label = get_batch_cls(train_set, self.train_step, batch_size, batch_index)
#                    sess.run(self.optimizer, feed_dict={self.x:train_data, self.y:train_label})
#
#                if epoch% self.conf['save_ckpt_epoch'] == 0:
#                    train_data, train_label = get_batch_cls(train_set, self.train_step, batch_size, 0)
#                    l2loss, train_sum =  sess.run([self.loss, self.train_summary], feed_dict={self.x:train_data, self.y:train_label})
#                    self.summary_writer.add_summary(train_sum, epoch)
#                    save_ckpt(self.saver, sess, self.conf['checkpoint_dir'], self.conf['ckpt_name'], epoch)
#                    pbar.set_description('train l2loss: {}'.format(l2loss))    
#				
#                if epoch% self.conf['evaluation_epoch']  == 0:
#                    val_data, val_label = get_batch_cls(test_set, self.train_step, batch_size, 0)
#                    l2loss, val_sum =  sess.run([self.loss_eval, self.test_summary], feed_dict={self.x: val_data, self.y: val_label})         
#                    self.summary_writer.add_summary(val_sum, epoch)
#                    eval_bar.update(self.conf['evaluation_epoch'])
#                    eval_bar.set_description('Eval l2loss: {}'.format(l2loss))
#                    
                    




