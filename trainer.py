import tensorflow as tf
import sessionWrapper as sesswrapper
import data_process as dp
import model_zoo as mz
import netFactory as nf
import numpy as np

filepath = './data/all_data.pkl'

train_step = 10
predict_step = 5
batch_size = 8

n_linear_hidden_units = 15
n_lstm_hidden_units = 15
n_attlstm_hidden_units = 15
n_att_hidden_units = 15

tv_gen = dp.train_validation_generaotr()
train, validation = tv_gen.generate_train_val_set(filepath, ['1101'], train_step, predict_step, 0.25, ['20130302', '20130902'])


x = tf.placeholder(tf.float32, [None, train_step, train.shape[-1]]) 
y = tf.placeholder(tf.float32, [None, predict_step]) 
isTrain = tf.placeholder(tf.bool, ())  

decoder_output = mz.baseline_LuongAtt_lstm(	x, batch_size, y, n_linear_hidden_units, n_lstm_hidden_units,
											n_attlstm_hidden_units, n_att_hidden_units, 
											isTrain, reuse = False)

loss = tf.losses.mean_squared_error(y, decoder_output)
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


with tf.name_scope('train_summary'):
    tf.summary.scalar('l2loss', loss, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')     
    
with tf.name_scope('validatin_summary'):
    tf.summary.scalar('l2loss', loss, collections=['validatin'])
    merged_summary_val = tf.summary.merge_all('validatin') 

sess = sesswrapper.sessionWrapper(	x, y, isTrain, train_step,
									loss, train_op, merged_summary_train, 
									merged_summary_val, './model/test', 'test.ckpt')

sess.run(batch_size, train, validation)

#def get_batch(data_set, train_step,batch_size, cur_index):
#    
#    #data_set: [None, time_step, features ]
#    #batch_idx: index of batch start point
#
#    batch =  data_set[cur_index:cur_index + batch_size, :, :]
#    train, label = np.split(batch, [train_step], axis=1)
#    label = label[:,:,-1]
#
#    return train, label
#
#reuse = False
#
#with tf.variable_scope('baseline') as scope:
#
#        if reuse: scope.reuse_variables()
#        train_data, train_label = get_batch(train, train_step, batch_size, 0)
#
#        with tf.name_scope('encoder') as scope:
#            encoder_output, final_state = nf.encoder(x, n_linear_hidden_units, n_lstm_hidden_units, batch_size)
#            decoder_cell = nf.attention_lstm_cell(encoder_output, n_attlstm_hidden_units, n_att_hidden_units)
#            #decoder_cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden_units)
#
#        with tf.name_scope('decoder') as scope:
#            decoder_output, _  = tf.cond(tf.equal(isTrain, tf.constant(True)), 
#									lambda: nf.decoder(decoder_cell,  encoder_output[:,-1], final_state, predict_step, train = True), 
#									lambda: nf.decoder(decoder_cell, encoder_output[:,-1], final_state, predict_step, train = False))

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    b = sess.run(decoder_output, feed_dict={x:train_data, y:train_label, isTrain:True})





