import sys
sys.path.append('/home/ubuntu/shared/workspace/etf_prediction')
import tensorflow as tf
import random
import numpy as np
import ray
from tqdm import tqdm

import hparam as conf
import data_process_list as dp
import model_zoo as mz
import loss_func as l


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
    label = label[:,:,56:]

    return train, label

tf.reset_default_graph()  

c = conf.config('test_onlyEnc_biderect_gru_cls').config['common']
#c['src_file_path'] = '../Data/all_feature_data.pkl'

tv_gen = dp.train_validation_generaotr()
if 'random' in c['sample_type']:  tv_gen.generate_train_val_set =  tv_gen.generate_train_val_set_random
train, validation = tv_gen.generate_train_val_set(c['src_file_path'], c['input_stocks'], c['input_step'], c['predict_step'], c['train_eval_ratio'], c['train_period'])
Ndata = len(train)


@ray.remote(num_gpus=1)
def Enc_gru_cls(conf, steps, train, validation, weights = None):
    
    c['input_step'] = conf['input_step']
    c['batch_size'] = conf['batch_size']
    c['n_lstm_hidden_units'] = conf['n_lstm_hidden_units']
    
    
    with tf.Graph().as_default():

        if c['feature_size'] == None: c['feature_size'] = train.shape[-1]
        #x = tf.placeholder(tf.float32, [None, c['input_step'], train.shape[-1]])
        x = tf.placeholder(tf.float32, [None, c['input_step'], c['feature_size']])
        y = tf.placeholder(tf.float32, [None, c['predict_step'], 3]) 
        
        decoder_output = mz.model_zoo(c, x, y, dropout = 0.6, is_train = True).decoder_output
        decoder_output_eval = mz.model_zoo(c, x, y, dropout = 1.0, is_train = False).decoder_output
        
        predict_train = tf.argmax(tf.nn.softmax(decoder_output), axis=-1)
        predict_eval = tf.argmax(tf.nn.softmax(decoder_output_eval), axis=-1)
        ground_truth = tf.argmax(y, axis=-1)
        accuracy_train = tf.reduce_mean(tf.cast(tf.equal(predict_train, ground_truth), tf.float32))
        accuracy_eval = tf.reduce_mean(tf.cast(tf.equal(predict_eval, ground_truth), tf.float32))
        
        l2_reg_loss = 0
        for tf_var in tf.trainable_variables():
                        #print(tf_var.name)
            if not ("bias" in tf_var.name or "output_project" in tf_var.name):
                l2_reg_loss +=  tf.reduce_mean(tf.nn.l2_loss(tf_var))
                
        
        loss = l.cross_entropy_loss(decoder_output, y) 
        loss_eval = l.cross_entropy_loss(decoder_output_eval, y)
        #train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        train_op = tf.train.RMSPropOptimizer(1e-2, 0.9).minimize(loss)
        
        Nbatch = Ndata//c['batch_size'] 
        
        with tf.Session() as sess:
             variables = ray.experimental.TensorFlowVariables(loss, sess)
             sess.run(tf.global_variables_initializer())
             
             if weights is not None:
                    variables.set_weights(weights)
             
             for i in range(1, steps + 1):
                 for j in range(Nbatch):
                     train_data, train_label = get_batch_random_cls(train, c['input_step'],  c['batch_size'] ,
                                                                    0,  c['feature_size'])
                     sess.run(train_op, feed_dict={x:train_data, y:train_label})
             
             val_data, val_label = get_batch_random_cls(validation, c['input_step'],  c['batch_size'] ,
                                                        0,  c['feature_size'])
             acc = sess.run(accuracy_eval, feed_dict={x:val_data, y:val_label})
            
        return acc
                 

if __name__ == "__main__":
    
    trials = 10
    steps = 1000
    
    ray.init()
    train_data = ray.put(train)
    
    best_hyperparameters = None
    best_accuracy = 0
    remaining_ids = []
    hyperparameters_mapping = {}
    
    
    
    def generate_hyperparameters():
        
        batch_set =list(range(32,128,16))
        input_step =list(range(20,100,30))
        n_lstm_hidden_units = list(range(64,256,64))
        
        return {"input_step": batch_set[np.random.randint(0, len(input_step))],
                "batch_size": batch_set[np.random.randint(0, len(batch_set))],
                "n_lstm_hidden_units": batch_set[np.random.randint(0, len(n_lstm_hidden_units))]
                }
        
    for i in tqdm(range(trials)):
        hyperparameters = generate_hyperparameters()
        acc_id = Enc_gru_cls.remote(hyperparameters, steps, train, validation)
        remaining_ids.append(acc_id)
        hyperparameters_mapping[acc_id] = hyperparameters
        
     # Fetch and print the results of the tasks in the order that they complete.
    for i in tqdm(range(trials)):
        # Use ray.wait to get the object ID of the first task that completes.
        ready_ids, remaining_ids = ray.wait(remaining_ids)
        # Process the output of this task.
        result_id = ready_ids[0]
        hyperparameters = hyperparameters_mapping[result_id]
        acc = ray.get(result_id)
        print("""We achieve accuracy {:.3}% with
            input_step: {}
            batch_size: {}
            n_lstm_hidden_units: {}
          """.format( acc,
                     hyperparameters["input_step"],
                     hyperparameters["batch_size"],
                     hyperparameters["n_lstm_hidden_units"]
                     ))
        if acc > best_accuracy:
            best_hyperparameters = hyperparameters
            best_accuracy = acc
            
    print("""Best accuracy over {} trials was {:.3} with
            input_step: {}
            batch_size: {}
            n_lstm_hidden_units: {}
          """.format( trials, acc,
                     best_hyperparameters["input_step"],
                     best_hyperparameters["batch_size"],
                     best_hyperparameters["n_lstm_hidden_units"]
                     ))

    
       


