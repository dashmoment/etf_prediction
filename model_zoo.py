import tensorflow as tf
import netFactory as nf

class model_zoo:
    
    def __init__(self, conf, inputs, y_label, is_train = True, dropout = 0.6):
        
        self.conf = conf
        self.inputs = inputs
        self.y_label = y_label
        self.is_train =  is_train   
        self.decoder_output = None  
        self.dropout = dropout
        
        
        self.get_config(self.conf['model'])
            
    def get_config(self, conf_name):

        try:
            conf = getattr(self, conf_name)
            conf()

        except: 
            print("Can not find configuration")
            raise
#==============================RNN===================================================
    def baseline_encReg_gru(self):
        
        def project_fn(tensor):
            
            output_projecter = tf.layers.Dense(5, name="output_project")  
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            d_layer = output_projecter(tensor)
                    
            return d_layer
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                #encoder_output, final_state = nf.encoder_GRU(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                cell = tf.contrib.rnn.DropoutWrapper(cell, self.dropout)
                init_state = cell.zero_state(self.conf['batch_size'], dtype=tf.float32) 
                encoder_output, final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state, time_major=False,  scope='lstm_encoder') 

                #decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units'])     
                encoder_output = tf.transpose(encoder_output, (1,0,2))
                print(encoder_output)
                self.decoder_output = project_fn(encoder_output[-1])


    def baseline_encReg_biderect_gru(self):
        
        def project_fn(tensor):
            
            output_projecter = tf.layers.Dense(5, name="output_project")  
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            d_layer = output_projecter(tensor)
                    
            return d_layer
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):
                #encoder_output, final_state = nf.encoder_GRU(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                fw_cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, self.dropout)
                bw_cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, self.dropout)
                
#                fw_init_state = fw_cell.zero_state(self.conf['batch_size'], dtype=tf.float32)
#                bw_init_state = bw_cell.zero_state(self.conf['batch_size'], dtype=tf.float32)  
                encoder_output,final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.inputs, dtype="float32", scope='gru_bidection')
                #encoder_output,final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.inputs, initial_state_fw=fw_init_state, initial_state_bw=bw_init_state, scope='gru_bidection')
                encoder_output = tf.concat(encoder_output, 2)   
                encoder_output = tf.transpose(encoder_output, (1,0,2))
                
                self.decoder_output = project_fn(encoder_output[-1])

    def baseline_encReg_stacked_gru(self):
        
        def project_fn(tensor):
            
            output_projecter = tf.layers.Dense(5, name="output_project")  
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            d_layer = output_projecter(tensor)
                    
            return d_layer
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):

            stack_layers = 2
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                #encoder_output, final_state = nf.encoder_GRU(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                
                cells = []
                
                for _ in range(stack_layers):

                    tmp_cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                    tmp_cell = tf.contrib.rnn.DropoutWrapper(tmp_cell, self.dropout)
                    cells.append(tmp_cell)

#                states = [cells[i].zero_state(self.conf['batch_size'], dtype=tf.float32) for i in range(stack_layers)]
                cells = tf.contrib.rnn.MultiRNNCell(cells)
                init_state = cells.zero_state(self.conf['batch_size'], dtype=tf.float32) 
                encoder_output, final_state = tf.nn.dynamic_rnn(cells, self.inputs, initial_state=init_state, time_major=False,  scope='stacked_gru') 
                encoder_output = tf.transpose(encoder_output, (1,0,2))
                
                self.decoder_output = project_fn(encoder_output[-1])


    def baseline_encReg_biderect_gru_cls(self):

#        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
            
#            l1 = tf.layers.Dense(16, name="output_project",  activation=tf.nn.relu)
#            l2 = tf.layers.Dense(16, name="output_project",  activation=tf.nn.relu)
#            l3 = tf.layers.Dense(3, name="output_project",  activation=None)
#            
#            net = l1(self.inputs)
#            net = l2(net)
#            self.decoder_output = l3(net)

#            cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
#            init_state = cell.zero_state(self.conf['batch_size'], dtype=tf.float32) 
#            encoder_output,final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
#            encoder_output = tf.transpose(encoder_output, (1,0,2))
#            output_projecter = tf.layers.Dense(15, name="output_project") 
#            self.decoder_output = output_projecter(encoder_output[-1]) 
#            self.decoder_output = tf.reshape(self.decoder_output, (-1,5,3))
            

        
        def project_fn(tensor):
            
            output_projecter = tf.layers.Dense(15, name="output_project")  
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            d_layer = output_projecter(tensor)
            d_layer = tf.reshape(d_layer, (-1, self.conf['predict_step'], 3))
                    
            return d_layer
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):
                
                fw_cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, self.dropout)
                bw_cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, self.dropout)
                encoder_output,final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.inputs, dtype="float32", scope='gru_bidection')
                
                encoder_output = tf.concat(encoder_output, 2)   
                encoder_output = tf.transpose(encoder_output, (1,0,2))
                
                self.decoder_output = project_fn(encoder_output[-1])
        
#==============================Seq2Seq===================================================           
        
    def baseline_LuongAtt_lstm(self):
        
        def project_fn(tensor):
            
            output_projecter = tf.layers.Dense(1, name="output_project")  
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            d_layer = output_projecter(tensor)
                    
            return d_layer
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units'])         
                #decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['n_lstm_hidden_units'])
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.dropout)
            
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):  
                
              
                if self.is_train:
                    self.decoder_output, _  = nf.decoder(self.conf['batch_size'], decoder_cell, project_fn, self.y_label , 
                                                         final_state, self.conf['predict_step'], dropout = self.dropout, is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder(self.conf['batch_size'], decoder_cell, project_fn, encoder_output[:,-1] , 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)

    def baseline_LuongAtt_gru(self):
        
        def project_fn(tensor):
            
            output_projecter = tf.layers.Dense(1, name="output_project")  
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            d_layer = output_projecter(tensor)
                    
            return d_layer
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder_GRU(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                #decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units'])     
                
                decoder_cell = tf.contrib.rnn.GRUCell(self.conf['n_lstm_hidden_units'])
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.dropout)
            
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):  
                
              
                if self.is_train:
                    self.decoder_output, _  = nf.decoder_GRU(self.conf['batch_size'], decoder_cell, project_fn, self.y_label , 
                                                         final_state, self.conf['predict_step'], dropout = self.dropout, is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder_GRU(self.conf['batch_size'], decoder_cell, project_fn, encoder_output[:,-1] , 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)

    
                    
                    
    def baseline_LuongAtt_lstm_cls(self):
        
        output_projecter = tf.layers.Dense(3, name="output_project")  
        
        def project_fn(tensor):
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            output = output_projecter(tensor)
            
            #if not self.is_train:
            #    output = tf.nn.softmax(output, name='softmax')

              
            return output
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units'])              
                #decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['n_lstm_hidden_units'])
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.dropout)
                       
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):  
                
        
                if self.is_train:
                    self.decoder_output, _  = nf.decoder_cls(self.conf['batch_size'], decoder_cell, project_fn, self.y_label , 
                                                         final_state, self.conf['predict_step'], dropout = self.dropout, is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder_cls(self.conf['batch_size'], decoder_cell, project_fn, encoder_output[:,-1] , 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)
    
    def baseline_LuongAtt_lstm_2in1(self):
        
        output_projecter = tf.layers.Dense(4, name="output_project")  
        
        def project_fn(tensor):
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            output = output_projecter(tensor)
            
            if not self.is_train:
                output_price = tf.slice(output, (0,0), (self.conf['batch_size'], 1))
                output_ud = tf.slice(output, (0,1), (self.conf['batch_size'], 3))
                output_ud = tf.nn.softmax(output_ud, name='softmax')
                output =tf.concat([output_price, output_ud], axis = -1)
              
            return output
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])               
                decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units'])                
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.dropout)
                #decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['n_lstm_hidden_units'])
            
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):  
                
        
                if self.is_train:
                    self.decoder_output, _  = nf.decoder_2in_1(self.conf['batch_size'], decoder_cell, project_fn, self.y_label , 
                                                         final_state, self.conf['predict_step'], dropout = self.dropout, is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder_2in_1(self.conf['batch_size'], decoder_cell, project_fn,  encoder_output[:,-1], 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)
                
    def baseline_LuongAtt_lstm_cnn_cls(self):
        
        output_projecter = tf.layers.Dense(3, name="output_project")  
        
        def project_fn(tensor):
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            output = output_projecter(tensor)
            
            if not self.is_train:
                output = tf.nn.softmax(output, name='softmax')

              
            return output
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
            
            cnn_encoder = nf.cnn_encoder_vgg(self.inputs, self.dropout, is_training = self.is_train)
            
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder(cnn_encoder,  self.conf['n_linear_hidden_units'], 
                                                self.conf['n_lstm_hidden_units'], self.conf['batch_size'], time_step= self.conf['input_step'], nFeatures=4)
            
                decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units'])      
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.dropout)
                #decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['n_lstm_hidden_units'])
            
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):  
                
        
                if self.is_train:
                    self.decoder_output, _  = nf.decoder_cls(self.conf['batch_size'], decoder_cell, project_fn, self.y_label , 
                                                         final_state, self.conf['predict_step'], dropout = self.dropout, is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder_cls(self.conf['batch_size'], decoder_cell, project_fn, encoder_output[:,-1] , 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)

        
