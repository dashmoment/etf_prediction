import tensorflow as tf
import netFactory as nf

class model_zoo:
    
    def __init__(self, conf, inputs, y_label, is_train, reuse=False):
        
        self.conf = conf
        self.inputs = inputs
        self.y_label = y_label
        self.is_train =  is_train
        self.reuse = reuse
        
        self.decoder_output = None
        
        self.get_config(self.conf['model'])
        
    def get_config(self, conf_name):

        try:
            conf = getattr(self, conf_name)
            conf()

        except: 
            print("Can not find configuration")
            raise
        
    def baseline_LuongAtt_lstm(self):
        
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
        
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                encoder_output, final_state = nf.encoder(self.inputs,  self.conf['n_linear_hidden_units'], self.conf['n_lstm_hidden_units'], self.conf['batch_size'])
                decoder_cell = nf.attention_lstm_cell(encoder_output, self.conf['n_lstm_hidden_units']) 
                #decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['n_lstm_hidden_units'])
            
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                
                if self.is_train:
                    self.decoder_output, _  = nf.decoder(self.conf['batch_size'], decoder_cell, self.y_label , final_state, self.conf['predict_step'], is_train = True)
                else:
                    self.decoder_output, _  = nf.decoder(self.conf['batch_size'], decoder_cell, encoder_output[:,-1] , final_state, self.conf['predict_step'], is_train = False)
  
    

