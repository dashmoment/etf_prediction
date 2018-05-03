import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def load_ckpt(saver, sess, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
#       
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)


        if ckpt  and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False 

def save_ckpt(saver, sess, checkpoint_dir, ckpt_name, step):
    
    print(" [*] Saving checkpoints...step: [{}]".format(step))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, 
               os.path.join(checkpoint_dir, ckpt_name),
               global_step=step)

def get_batch(data_set, train_step,batch_size, cur_index):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    label = label[:,:,3]

    return train, label

def get_batch_cls(data_set, train_step,batch_size, cur_index):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
   
    label = label[:,:,10:13]

    return train, label

def get_batch_2in1(data_set, train_step,batch_size, cur_index):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    label_ud = label[:,:,10:13]
    label_p = label[:,:,3]
    
    label = np.dstack([label_p,label_ud])

    return train, label


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
                              '2in1':get_batch_2in1}

    def run(self, train_set, test_set):
        
        get_batch = self.sample_method [self.conf['sample_type']]
        
        epoch =  self.conf['current_epoch']
        pbar = tqdm(total =  self.conf['total_epoch'])
        pbar.update(epoch)
        eval_bar = tqdm(total =  self.conf['total_epoch'])
        eval_bar.update(epoch)
        
        batch_size =  self.conf['batch_size']
        Nbatch = len(train_set)//batch_size
        
        with self.sess as sess:
            
            if load_ckpt(self.saver, sess, self.conf['checkpoint_dir'], self.conf['ckpt_name']):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            while epoch <  self.conf['total_epoch']:
                epoch += 1
                pbar.update(1)
                np.random.shuffle(train_set)
                
                #Cehck variable reused
#                tvars = tf.trainable_variables()
#                tvars_vals = sess.run(tvars)
#                for var, val in zip(tvars, tvars_vals):
#                    print(var.name)
#                break

                for i in range(Nbatch):
                    batch_index = i*batch_size
                    train_data, train_label = get_batch(train_set, self.train_step, batch_size, batch_index)
                    sess.run(self.optimizer, feed_dict={self.x:train_data, self.y:train_label})

                if epoch% self.conf['save_ckpt_epoch'] == 0:
                    train_data, train_label = get_batch(train_set, self.train_step, batch_size, 0)
                    l2loss, train_sum =  sess.run([self.loss, self.train_summary], feed_dict={self.x:train_data, self.y:train_label})
                    self.summary_writer.add_summary(train_sum, epoch)
                    save_ckpt(self.saver, sess, self.conf['checkpoint_dir'], self.conf['ckpt_name'], epoch)
                    pbar.set_description('train l2loss: {}'.format(l2loss))    
				
                if epoch% self.conf['evaluation_epoch']  == 0:
                    val_data, val_label = get_batch(test_set, self.train_step, batch_size, 0)
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
                    



