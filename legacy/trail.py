import tensorflow as tf
import numpy as np
import math
import random

train = []
validation = [] 


#for i in range(38500):
for i in range(0, 38500, 35):
    
    rand = random.random() * 2 * math.pi
    if i < 35000:
#        if rad >= np.pi*2:
#            rad = 0
#        else:
#            rad += 0.3
#        train.append([np.sin(rand), np.sin(rand)])
        temp_sig = np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, 35)
        train.append([temp_sig, np.sin(temp_sig)])        
    else:
#        if rad >= np.pi*2:
#            rad = 0
#        else:
#            rad += 0.5
#        validation.append([np.sin(rand), np.sin(rand)])
        temp_sig = np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, 35)
        validation.append([temp_sig, np.sin(temp_sig)])                
    
train = np.reshape(np.array(train), (-1,35,2))
validation = np.reshape(np.array(validation), (-1,35,2))


tf.nn.seq2seq = tf.contrib.legacy_seq2seq

input_dim = 1
seq_length = 30

with tf.variable_scope('Seq2seq'):

    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
           for t in range(seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="expected_sparse_output_".format(t))
          for t in range(5)
    ]
    
    # Give a "GO" token to the decoder. 
    # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ enc_inp[:-1]"
    dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") ] + enc_inp[:-1]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
    cells = []
    for i in range(2):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(16))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def. 
    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp, 
        dec_inp, 
        cell
    )
    
    # For reshaping the output dimensions of the seq2seq RNN: 
    w_out = tf.Variable(tf.random_normal([16, input_dim]))
    b_out = tf.Variable(tf.random_normal([input_dim]))
    
    # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    
reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out) for i in dec_outputs]


with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
        
    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            
    loss = output_loss  + 0.003*reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(0.007, decay=0.9, momentum = 0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()

def train_batch(data_set,train_step,batch_size, cur_index):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
#    print(np.shape(train), np.shape(label))
    
    train = train[:,:,0]
    label = label[:,:,1]
    
    feed_dict = {enc_inp[t]: np.reshape(train[:,t], [-1,1] ) for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: np.reshape(label[:,t], [-1, 1]) for t in range(5)})
    
    _, loss_t = sess.run([train_op, loss], feed_dict)
    
    return loss_t


def test_batch(data_set,train_step,batch_size, cur_index):
    
    #data_set: [None, time_step, features ]
    #batch_idx: index of batch start point

    batch =  data_set[cur_index:cur_index + batch_size, :, :]
    train, label = np.split(batch, [train_step], axis=1)
    
    train = train[:,:,0]
    label = label[:,:,1]
    
    feed_dict = {enc_inp[t]: np.reshape(train[:,t], [-1,1] ) for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: np.reshape(label[:,t], [-1, 1]) for t in range(5)})
    
    loss_t = sess.run([loss], feed_dict)
    
    return loss_t[0]


train_losses = []
test_losses = []
batch = 0

sess.run(tf.global_variables_initializer())
for t in range(20000):
    if batch > 4:batch = 0
    else: batch+=16
    train_loss = train_batch(train, 30, 16, batch)
    train_losses.append(train_loss)
    
    if t % 10 == 0: 
        # Tester
        test_loss = test_batch(validation, 30, 16, batch)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, 500, train_loss, test_loss))

print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))

#def generate_x_y_data_v1(isTrain, batch_size):
#    """
#    Data for exercise 1.
#    returns: tuple (X, Y)
#        X is a sine and a cosine from 0.0*pi to 1.5*pi
#        Y is a sine and a cosine from 1.5*pi to 3.0*pi
#    Therefore, Y follows X. There is also a random offset
#    commonly applied to X an Y.
#    The returned arrays are of shape:
#        (seq_length, batch_size, output_dim)
#        Therefore: (10, batch_size, 2)
#    For this exercise, let's ignore the "isTrain"
#    argument and test on the same data.
#    """
#    seq_length = 30
#
#    batch_x = []
#    batch_y = []
#    for _ in range(batch_size):
#        rand = random.random() * 2 * math.pi
#
#        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
#                                  3.0 * math.pi + rand, seq_length * 2))
#        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
#                                  3.0 * math.pi + rand, seq_length * 2))
#        x1 = sig1[:seq_length]
#        y1 = sig1[seq_length:]
#        x2 = sig2[:seq_length]
#        y2 = sig2[seq_length:]
#
#        x_ = np.array([x1, x2])
#        y_ = np.array([y1, y2])
#        x_, y_ = x_.T, y_.T
#
#        batch_x.append(x_)
#        batch_y.append(y_)
#
#    batch_x = np.array(batch_x)
#    batch_y = np.array(batch_y)
#    # shape: (batch_size, seq_length, output_dim)
#
#    batch_x = np.array(batch_x).transpose((1, 0, 2))
#    batch_y = np.array(batch_y).transpose((1, 0, 2))
#    # shape: (seq_length, batch_size, output_dim)
#
#    return batch_x, batch_y
#
#def train_batch(batch_size):
#    """
#    Training step that optimizes the weights 
#    provided some batch_size X and Y examples from the dataset. 
#    """
#    X, Y = generate_x_y_data_v1(isTrain=True, batch_size=batch_size)
#    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
#    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
#    _, loss_t = sess.run([train_op, loss], feed_dict)
#    return loss_t
#
#def test_batch(batch_size):
#    """
#    Test step, does NOT optimizes. Weights are frozen by not
#    doing sess.run on the train_op. 
#    """
#    X, Y = generate_x_y_data_v1(isTrain=False, batch_size=batch_size)
#    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
#    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
#    loss_t = sess.run([loss], feed_dict)
#    return loss_t[0]
#
#
#
#train_losses = []
#test_losses = []
#batch = 0
#
#sess.run(tf.global_variables_initializer())
#for t in range(20000):
#    if batch > 4:batch = 0
#    else: batch+=16
#    train_loss = train_batch(16)
#    train_losses.append(train_loss)
#    
#    if t % 10 == 0: 
#        # Tester
#        test_loss = test_batch(16)
#        test_losses.append(test_loss)
#        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, 500, train_loss, test_loss))
#
#print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
