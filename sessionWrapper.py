import tensorflow as tf
import os
from tqdm import tqdm

def save_ckpt(saver:tf.train.Saver(), sess, checkpoint_dir, ckpt_name, step):
    
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
		self.optimizer = optimizer
		self.train_summary = train_summary
		self.test_summary = test_summary

	def run(self, batch_size, train_set, test_set, epoch = 0, Nepoch = 1000):

		pbar = tqdm(total=Nepoch)
		Nbatch = len(train_set)//batch_size

		while epoch < Nepoch:
			epoch += 1
			np.random.shuffle(train_set)

			for i in range(Nbatch):         
	            batch_index = i*batch_size
	            train_data, train_label = get_batch(train_set, train_step, batch_size, batch_index)
	            self.sess.run(train_op, feed_dict={self.x:train_data, self.y:train_label, self.is_train:True})

	        if epoch%11 == 0:

	        	l2loss, train_sum =  self.sess.run([loss, merged_summary_train], feed_dict={self.x:train_data, self.y:label, self.is_train:False})
	        	self.summary_writer.add_summary(train_sum, epoch)
	        	save_ckpt(self.saver, self.sess, self.checkpoint_dir, self.ckpt_name, epoch)
	        	pbar.set_description('train l2loss: {}'.format(l2loss))    

	        elif epoch%20 == 0:

	        	val_data, val_label = get_batch(test_set, train_step, batch_size, 0)
           		l2loss, val_sum =  self.sess.run([self.loss, self.test_summary], feed_dict={self.x: val_data, self.y: val_label, self.is_train:False})         
            	self.summary_writer.add_summary(val_sum, epoch)
            	pbar.set_description('val l2loss: {}'.format(l2loss))




