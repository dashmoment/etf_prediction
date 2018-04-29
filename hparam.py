class config:

    def __init__(self, configuration):
		
        self.configuration = configuration
        self.config = {
						"common":{},
						}
        self.get_config()


    def get_config(self):

        try:
            conf = getattr(self, self.configuration)
            conf()

        except: 
            print("Can not find configuration")
            raise
            
    def sample(self):
        
        conf = self.config['common']   
        conf['src_file_path'] = './Data/all_data.pkl'
        conf['checkpoint_dir'] = './model/test'
        conf['ckpt_name'] = 'test.ckpt'
        conf['input_step'] = 10
        conf['predict_step'] = 5
        conf['batch_size'] = 8
        conf['train_eval_ratio'] = 0.25 
        conf['n_linear_hidden_units'] = 15
        conf['n_lstm_hidden_units'] = 12
        conf['n_attlstm_hidden_units'] = 12
        conf['n_att_hidden_units'] = 18
        conf['train_period'] =  ['20130302', '20130404']
        conf['eval_period'] =  ['20140302', '20140404']