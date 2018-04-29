import tensorflow as tf
import netFactory as nf

def baseline_LuongAtt_lstm( inputs, batch, train_y, n_linear_hidden_units, n_lstm_hidden_units, 
							n_attlstm_hidden_units, n_att_hidden_units, 
							is_train, reuse = False, predict_time_step = 5):
    
    
    with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
        with tf.name_scope('encoder'):
            encoder_output, final_state = nf.encoder(inputs,  n_linear_hidden_units, n_lstm_hidden_units, batch)
            decoder_cell = nf.attention_lstm_cell(encoder_output, n_attlstm_hidden_units, n_att_hidden_units) 
            print(encoder_output[:,-1], train_y[:,-1])
			#decoder_cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden_units)
           

        with tf.name_scope('decoder'):
            
            decoder_output, _  = tf.cond(tf.equal(is_train, tf.constant(True)), 
    									lambda: nf.decoder(batch, decoder_cell,  train_y, final_state, predict_time_step, is_train = True), 
    									lambda: nf.decoder(batch, decoder_cell, encoder_output[:,-1], final_state, predict_time_step, is_train = False))
			
    return decoder_output


