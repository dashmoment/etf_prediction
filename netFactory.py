import tensorflow as tf
import numpy as np

def max_pool_layer(inputs, kernel_shape, stride, name=None, padding='VALID'):
          
    return tf.nn.max_pool(inputs, kernel_shape, stride, padding, name=name)

def avg_pool_layer(inputs, kernel_shape, stride, name=None, padding='VALID'):
          
    return tf.nn.avg_pool(inputs, kernel_shape, stride, padding, name=name)
    
def lrelu(x, name = "leaky", alpha = 0.2):

    with tf.variable_scope(name):
        leaky = tf.maximum(x, alpha * x, name=name)
        #leaky = tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    return leaky

def batchnorm(input, index = 0, reuse = False):
    with tf.variable_scope("batchnorm_{}".format(index), reuse = reuse):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    return normalized

def convolution_layer(inputs, kernel_shape, stride, name, pre_shape = None, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu, is_bn=False):
                                                                                            #initializer=tf.contrib.layers.xavier_initializer()
    
    if pre_shape == None: pre_shape = inputs.get_shape()[-1]
    rkernel_shape = [kernel_shape[0], kernel_shape[1], pre_shape, kernel_shape[2]]     
    
    with tf.variable_scope(name) as scope:
        
        try:
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        
        net = tf.nn.conv2d(inputs, weight,stride, padding=padding)
        net = tf.add(net, bias)

        if is_bn:
            net = batchnorm(net)
        else:
            net = net
        
        if not activat_fn==None:
            net = activat_fn(net, name=name+"_out")
        
        if flatten == True:
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
        
    return net

def fc_layer(inputs, out_shape, name,initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    pre_shape = inputs.get_shape()[-1]
    
    with tf.variable_scope(name) as scope:
        
        
        try:
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        
        
        if activat_fn != None:
            net = activat_fn(tf.nn.xw_plus_b(inputs, weight, bias, name=name + '_out'))
        else:
            net = tf.nn.xw_plus_b(inputs, weight, bias, name=name)
        
    return net


def encoder(inputs, n_linear_hidden_units, n_lstm_hidden_units, batch, time_step = None, nFeatures = None):
	
    if time_step == None:
        _, timeStep, nFeatures = inputs.shape
    
    else:
        timeStep = time_step
        nFeatures = nFeatures
    
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

def encoder_GRU(inputs, n_linear_hidden_units, n_lstm_hidden_units, batch, time_step = None, nFeatures = None):
    
    if time_step == None:
        _, timeStep, nFeatures = inputs.shape
    
    else:
        timeStep = time_step
        nFeatures = nFeatures
    
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
    cell = tf.contrib.rnn.GRUCell(n_lstm_hidden_units)
    init_state = cell.zero_state(batch, dtype=tf.float32) 
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, time_major=False,  scope='lstm_encoder') 

    # output: [batch, timeStep, cell.nFeatures] 
    # final_state: [batch_size, cell.state_size]
    
    return outputs, final_state


def decoder_GRU(batch, decoder_cell, project_fn, previous_y, state, predict_time_step,  dropout = 0.6, is_train=True): 

    def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
        return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

    def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_time_step
    
    


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


    if is_train:
                        
        loop_init_train =   [   tf.constant(0, dtype=tf.int32), #time
                                tf.reshape(previous_y[:,0], (-1, 1)), 
                                state, 
                                #decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
                                tf.TensorArray(dtype=tf.float32, size=predict_time_step),
                                    tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_train)

    else:      
  
        loop_init_inference = [ tf.constant(0, dtype=tf.int32), #time
                                    project_fn(previous_y),
                                state, 
                                #decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
                                tf.TensorArray(dtype=tf.float32, size=predict_time_step),
                                    tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_inference)
        
    targets = targets_ta.stack()
    targets = tf.squeeze(targets, axis=-1)
    targets = tf.transpose(targets, (1,0))

    raw_outputs = outputs_ta.stack()
    raw_outputs = tf.transpose(raw_outputs, (1,0,2))

    return targets, raw_outputs

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


def decoder(batch, decoder_cell, project_fn, previous_y, state, predict_time_step,  dropout = 0.6, is_train=True): 

    def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
        return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

    def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_time_step
    
    


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


    if is_train:
                        
        loop_init_train =   [	tf.constant(0, dtype=tf.int32), #time
    	                    	tf.reshape(previous_y[:,0], (-1, 1)), 
    	                    		#state, 
                                decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_train)

    else:      
  
        loop_init_inference = [	tf.constant(0, dtype=tf.int32), #time
    	                    		project_fn(previous_y),
                                #state, 
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


def decoder_cls(batch, decoder_cell, project_fn, previous_y, state, predict_time_step,  dropout = 0.6, is_train=True): 

    def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
        return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

    def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_time_step


    def loop_fn_train(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):

        
        next_input = tf.reshape(previous_y[:,time], (-1, 3))
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
        
        return time + 1, tf.nn.softmax(projected_output), state, array_targets, array_outputs


    if is_train:
        
        #decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state)
        loop_init_train =   [	tf.constant(0, dtype=tf.int32), #time
    	                    		tf.reshape(previous_y[:,0], (-1, 3)), 
    	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_train)

    else:      
  
        loop_init_inference = [	tf.constant(0, dtype=tf.int32), #time
    	                    		tf.nn.softmax(project_fn(previous_y)), 
    	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_inference, loop_init_inference)
        
    targets = targets_ta.stack()
#    targets = tf.squeeze(targets, axis=-1)
    targets = tf.transpose(targets, (1,0,2))

    raw_outputs = outputs_ta.stack()
    raw_outputs = tf.transpose(raw_outputs, (1,0,2))

    return targets, raw_outputs

def decoder_2in_1(batch, decoder_cell, project_fn, previous_y, state, predict_time_step,  dropout = 0.6, is_train=True): 

    def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
        return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

    def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_time_step


    def loop_fn_train(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):

        #next_input = previous_y[:,time]
        next_input = tf.reshape(previous_y[:,time], (-1, 1))
        next_input = prev_output
        output, state = decoder_cell(next_input, prev_state)
        projected_output = project_fn(output)

        array_outputs = array_outputs.write(time, output)
        array_targets = array_targets.write(time, projected_output)
        
        return time + 1, projected_output, state, array_targets, array_outputs

    def softmax_out(tensor):
        y_price = tf.slice(tensor, [0,0], [batch, 1])
        y_ud = tf.nn.softmax(tf.slice(tensor, [0,1], [batch, 3]))
        return tf.concat([y_price, y_ud],-1)


    def loop_fn_inference(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):

        next_input = prev_output
        output, state = decoder_cell(next_input, prev_state)
        projected_logits = project_fn(output)

        projected_output = softmax_out(projected_logits)

        array_outputs = array_outputs.write(time, output)
        array_targets = array_targets.write(time, projected_logits)
        
        return time + 1, projected_output, state, array_targets, array_outputs


    if is_train:
                        
        loop_init_train =   [	tf.constant(0, dtype=tf.int32), #time
    	                    		tf.reshape(previous_y[:,0], (-1, 4)), 
    	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_train, loop_init_train)

    else:      
        
        loop_init_inference = [	tf.constant(0, dtype=tf.int32), #time
    	                    		softmax_out(project_fn(previous_y)), 
    	                    		decoder_cell.zero_state(batch, tf.float32).clone(cell_state=state),
    	                   		tf.TensorArray(dtype=tf.float32, size=predict_time_step),
    	                    		tf.TensorArray(dtype=tf.float32, size=predict_time_step) ]
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn_inference, loop_init_inference)

        
    targets = targets_ta.stack()
    targets = tf.transpose(targets, (1,0,2))

    raw_outputs = outputs_ta.stack()
    raw_outputs = tf.transpose(raw_outputs, (1,0,2))

    return targets, raw_outputs


def cnn_encoder_vgg(inputs, dropout, is_training):
    
    #inputs: [batch, times, features, stocks]
    
     with tf.variable_scope("cnn_encoder", reuse=tf.AUTO_REUSE):
                    
            net = convolution_layer(inputs, [1,3,128], [1,1,1,1],name="conv2-1")        
            net = tf.nn.max_pool(net, ksize=[1, 1, 3, 1],strides=[1, 1, 2, 1], padding='SAME')       
            net = convolution_layer(net, [1,3,1], [1,1,1,1],name="conv4-2")
          
            
     return net
                  
                    
    



