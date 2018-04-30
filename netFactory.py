import tensorflow as tf

def encoder(inputs, n_linear_hidden_units, n_lstm_hidden_units, batch):
	
    _, timeStep, nFeatures = inputs.shape
    weight =  tf.get_variable("w_encoder", [nFeatures, n_linear_hidden_units], initializer = tf.zeros_initializer())
    biases =  tf.get_variable('b_encoder', [n_linear_hidden_units, ], initializer = tf.constant_initializer(0.1))
    
                                             
    # hidden layer for input to cell  
    ########################################  
  
    # linear projection
    # input ==> (batch * steps, inputs)  
    inputs = tf.reshape(inputs, [-1, nFeatures])      
    inputs = tf.matmul(inputs, weight) + biases 
    inputs = tf.reshape(inputs, [-1, timeStep, n_linear_hidden_units]) 
    

    # lstm
    cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden_units)
    init_state = cell.zero_state(batch, dtype=tf.float32) 
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, time_major=False,  scope='lstm_encoder') 

    # output: [batch, timeStep, cell.nFeatures] 
    # final_state: [batch_size, cell.state_size]
    
    return outputs, final_state

def attention_lstm_cell(memory, n_lstm_hidden_units, att_type = 'Luong'):

	cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden_units)

	if att_type == 'Luong':
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(n_lstm_hidden_units, memory)
	else:
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(n_lstm_hidden_units, memory)

	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(	cell,
            												attention_mechanism,
       													attention_layer_size=n_lstm_hidden_units)

	return decoder_cell


def decoder(batch, decoder_cell, previous_y, state, predict_time_step, is_train=True):
    
    
    dense_for_project = tf.layers.Dense(1, name="output_project")

    def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
        return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

    def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_time_step
    
    def project_fn(tensor):
        d_layer =dense_for_project(tensor)
        return d_layer


    def loop_fn_train(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):

        #next_input = previous_y[:,time]
        next_input = tf.reshape(previous_y[:,time], (-1, 1))
        next_input = prev_output
        output, state = decoder_cell(next_input, prev_state)
        projected_output = project_fn(output)

        array_outputs = array_outputs.write(time, output)
        array_targets = array_targets.write(time, projected_output)
        
        return time + 1, projected_output, state, array_targets, array_outputs


    def loop_fn_inference(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):

        next_input = prev_output
        output, state = decoder_cell(next_input, prev_state)
        projected_output = project_fn(output)

        array_outputs = array_outputs.write(time, output)
        array_targets = array_targets.write(time, projected_output)
        
        return time + 1, projected_output, state, array_targets, array_outputs
    
    
#    def loop_init_train():
#        return  [	tf.constant(0, dtype=tf.int32), #time
#	                    		tf.reshape(previous_y[:,0], (-1, 1)), 
#	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
#	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
#	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
#        
#    def loop_init_inference():
#        return  [	tf.constant(0, dtype=tf.int32), #time
#	                    		project_fn(previous_y), 
#	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
#	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
#	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]


    if is_train:
#        decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state)
        
                  
        loop_init_train =   [	tf.constant(0, dtype=tf.int32), #time
    	                    		tf.reshape(previous_y[:,0], (-1, 1)), 
    	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_train)

    else:
        
  
        loop_init_inference = [	tf.constant(0, dtype=tf.int32), #time
    	                    		project_fn(previous_y), 
    	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_inference)
        
   

    targets = targets_ta.stack()
    targets = tf.squeeze(targets, axis=-1)
    targets = tf.transpose(targets, (1,0))

    raw_outputs = outputs_ta.stack()
    raw_outputs = tf.transpose(raw_outputs, (1,0,2))

    return targets, raw_outputs






