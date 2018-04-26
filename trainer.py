import tensorflow as tf
import sessionWrapper as sesswrapper
import data_process as dp
import model_zoo as mz

filepath = './Data/all_data.pkl'

train_step = 10
predict_step = 5
batch_size = 16

n_linear_hidden_units = 10
n_lstm_hidden_units = 12
n_attlstm_hidden_units = 14
n_att_hidden_units = 16

tv_gen = train_validation_generaotr()
train, validation = tv_gen.generate_train_val_set(filepath, ['1101'], 10, 5, 0.25, ['20130302', '20130502'])


x = tf.placeholder(tf.float32, [None, train_step, train.shape[-1]]) 
y = tf.placeholder(tf.float32, [None, test_step]) 
isTrain = tf.placeholder(tf.bool, ())  

decoder_output = mz.baseline_LuongAtt_lstm(	x, y, n_linear_hidden_units, n_lstm_hidden_units,
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

sess = sessionWrapper(	x, y, isTrain, train_step,
						loss, train_op, merged_summary_train, merged_summary_val)

sess.run(batch_size, train, validation)


