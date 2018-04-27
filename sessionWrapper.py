import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def load_ckpt(self, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
#        if ckpt_name == "":
#            model_dir = 'anonymous'
#        else:
#            model_dir = ckpt_name
#            
#        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(checkpoint_dir, ckpt.model_checkpoint_path)
        if ckpt  and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
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

    assert(len(data_set) < cur_index+batch_size, "batch index out of range")
    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    label = label[:,:,-1]

    return train, label

class sessionWrapper:

    def __init__(	self, input_p:tf.placeholder, label_p:tf.placeholder, is_train:tf.placeholder, train_step,
					loss, train_op, train_summary, test_summary,
					checkpoint_dir, ckpt_name):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'log'), self.sess.graph)
        
        self.x = input_p #tf.placeholder(tf.float32, [None, train_step, train_set.shape[-1]]) 
        self.y = label_p #tf.placeholder(tf.float32, [None, test_step])  
        self.is_train = is_train
        self.saver = tf.train.Saver()
        self.loss = loss
        self.optimizer = train_op
        self.train_step = train_step
        self.train_summary = train_summary
        self.test_summary = test_summary
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_name = ckpt_name

    def run(self, batch_size, train_set, test_set, epoch = 0, Nepoch = 1000):

        pbar = tqdm(total=Nepoch)
        Nbatch = len(train_set)//batch_size
        with self.sess as sess:
            
            if load_ckpt(self.checkpoint_dir, self.ckpt_name):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            while epoch < Nepoch:
                epoch += 1
                pbar.update(1)
                np.random.shuffle(train_set)

                for i in range(Nbatch):
                    batch_index = i*batch_size
                    train_data, train_label = get_batch(train_set, self.train_step, batch_size, batch_index)
                    sess.run(self.optimizer, feed_dict={self.x:train_data, self.y:train_label, self.is_train:True})

                if epoch%11 == 0:
                    train_data, train_label = get_batch(train_set, self.train_step, batch_size, 0)
                    l2loss, train_sum =  sess.run([self.loss, self.train_summary], feed_dict={self.x:train_data, self.y:train_label, self.is_train:False})
                    self.summary_writer.add_summary(train_sum, epoch)
                    save_ckpt(self.saver, sess, self.checkpoint_dir, self.ckpt_name, epoch)
                    pbar.set_description('train l2loss: {}'.format(l2loss))    
				
                elif epoch%20 == 0:
                    val_data, val_label = get_batch(test_set, self.train_step, batch_size, 0)
                    l2loss, val_sum =  sess.run([self.loss, self.test_summary], feed_dict={self.x: val_data, self.y: val_label, self.is_train:False})         
                    self.summary_writer.add_summary(val_sum, epoch)
                    pbar.set_description('val l2loss: {}'.format(l2loss))




