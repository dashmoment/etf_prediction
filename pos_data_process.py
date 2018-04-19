import numpy as np
import pandas as pd
import pickle

f = open('./Data/all_data.pkl', 'rb')
_ = pickle.load(f)
_ = pickle.load(f)
process_data = pd.DataFrame(pickle.load(f))

#select date
mask = (process_data.columns >= '20130102') & (process_data.columns < '20180302')
select_date = process_data.iloc[:,mask]

#drop NA
select_date = select_date.dropna()

for c in select_date.columns:    
    for idx in select_date[c].index:
        select_date[c].loc[idx] = select_date[c].loc[idx].drop(['ID', 'Date','name','trade']).tolist()
        
#make a stck_df to np_array 
stock_1101 = select_date.loc['1101']
stock_1101_np = np.vstack(np.array(stock_1101))

#make multiple stck_df to np_array        
stocks = ['1101','1102']
stock_1101_02 = select_date.loc[stocks]
stock_1101_02_np = np.hstack(np.array(stock_1101_02))
stock_1101_02_np = np.vstack(np.array(stock_1101_02_np))
stock_1101_02_np_r = np.split(stock_1101_02_np, len(stocks))
stock_1101_02_np_r = np.dstack(stock_1101_02_np_r)

#Check process is correct
print("single stock == multiple_stock: ",np.equal(stock_1101_np, stock_1101_02_np_r[:,:,0]).all())

#Prepare for rnn

import tensorflow as tf
import random

#Single stock

data = stock_1101_np
train_step = 12
test_step = 5
all_step = train_step + test_step
data_size = 50

data_all = np.zeros((data_size, all_step, data.shape[-1]))

for i in range(data_size):
    random_time_step = random.randint(0, len(data) - all_step)
    
    tmp = data[random_time_step:random_time_step+all_step,:]
    data_all[i] = tmp

# tf Graph input  
x = tf.placeholder(tf.float32, [None, train_step, data.shape[-1]])  


n_inputs = data.shape[-1]
n_hidden_units = 10
n_output = 3
batch_size = 16

def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
    return layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

# Define weights  
weights = {  
    # (28, 128)  
    'in': tf.Variable(tf.random_normal([data.shape[-1], n_hidden_units])),  
    # (128, 10)  
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_output]))  
}  
biases = {  
    # (128, )  
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  
    # (10, )  
    'out': tf.Variable(tf.constant(0.1, shape=[n_output, ]))  
}  

def encoder(X, weights, biases):  
    # hidden layer for input to cell  
    ########################################  
  
    # transpose the inputs shape from  
    # X ==> (128 batch * 28 steps, 28 inputs)  
    X = tf.reshape(X, [-1, n_inputs])  
  
    # into hidden  
    # X_in = (batch * time_steps, n_hidden_units)  
    X_in = tf.matmul(X, weights['in']) + biases['in']  
    # X_in ==> (batch, 28 time_steps, n_hidden_units)  
    X_in = tf.reshape(X_in, [-1, train_step, n_hidden_units])  
    
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,reuse=True )
    init_state = cell.zero_state(batch_size, dtype=tf.float32) 
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)  
    
    return outputs, final_state


def decoder(x, state, pred_input, previous_y):

    predict_days = 5
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,reuse=True )
    
    def cin_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_days
    
    def project_fn(tensor):
        return tf.layers.dense(tensor, 1, name='decoder_output_proj', kernel_initializer=default_init(0))

    def loop_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        
        next_input = prev_output
        output, state = cell(next_input, prev_state)
        projected_output = project_fn(output)
        
        array_outputs = array_outputs.write(time, output)
        array_targets = array_targets.write(time, projected_output)
        
        return time + 1, projected_output, state, array_targets, array_outputs
        
                                          
init = tf.global_variables_initializer()  
with tf.Session() as sess:  
    
    sess.run(init)
    np.random.shuffle(data_all)
    train, test = np.split(data_all, [train_step], axis=1)
    train, _ = np.split(train, [batch_size])
    
    out, state = sess.run(final_outputs, feed_dict={x:train})
    
    
    































