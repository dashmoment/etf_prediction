import tensorflow as tf
import netfactory as nf

def baseline_LuongAtt_lstm( input, train_y, n_linear_hidden_units, n_lstm_hidden_units, 
							n_attlstm_hidden_units, n_att_hidden_units, 
							train = True, reuse = False, predict_time_step = 5):
	
	with tf.variable_scope('baseline') as scope:

		if reuse: scope.reuse_variables()

		with tf.name_scope('encoder') as scope:
			encoder_output, final_state = nf.encoder(input, n_linear_hidden_units, n_lstm_hidden_units)
			decoder_cell = nf.attention_lstm_cell(encoder_output, n_attlstm_hidden_units, n_att_hidden_units)

		with tf.name_scope('decoder') as scope:
			if train:
				decoder_output, _ = nf.decoder(decoder_cell, train_y, final_state, predict_time_step, train = True)
			else:
				decoder_output, _ = nf.decoder(decoder_cell, encoder_output[:,-1], final_state, predict_time_step, train = False)

	return decoder_output


