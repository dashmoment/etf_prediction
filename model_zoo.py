import tensorflow as tf
import netFactory as nf

class model_zoo:
    
    def __init__(self, conf, inputs, y_label, is_train, dropout = 0.6):
        
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
                
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.dropout)
                #decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['n_lstm_hidden_units'])
            
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):  
                
              
                if self.is_train:
                    self.decoder_output, _  = nf.decoder(self.conf['batch_size'], decoder_cell, project_fn, self.y_label , 
                                                         final_state, self.conf['predict_step'], dropout = self.dropout, is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder(self.conf['batch_size'], decoder_cell, project_fn, encoder_output[:,-1] , 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)
                    
                    
    def baseline_LuongAtt_lstm_cls(self):
        
        output_projecter = tf.layers.Dense(3, name="output_project")  
        
        def project_fn(tensor):
        
            if self.is_train:
                tensor = tf.nn.dropout(tensor, keep_prob=self.dropout)
            output = output_projecter(tensor)
            
            if not self.is_train:
                output = tf.nn.softmax(output, name='softmax')

              
            return output
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
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
                    self.decoder_output, _  = nf.decoder_2in_1(self.conf['batch_size'], decoder_cell, project_fn, encoder_output[:,-1] , 
                                                         final_state, self.conf['predict_step'], dropout = 1.0, is_train = False)
                    
                    
    
    

