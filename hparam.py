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
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.15 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 32
        conf['train_period'] =  ['20130102', '20170102']
        conf['eval_period'] =  ['20170102', '20180311']