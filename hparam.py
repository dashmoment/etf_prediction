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
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['1101']
        conf['train_period'] =  ['20130102', '20140502']
        conf['eval_period'] =  ['20170102', '20180311']
        
        #Model Attributes
        conf['sample_type'] = 'reg'
        conf['model'] = 'baseline_LuongAtt_lstm'
        conf['checkpoint_dir'] = './model/test'
        conf['ckpt_name'] = 'test.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 1
        conf['train_eval_ratio'] = 0.15 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 16
        
        #Session Control
        conf['save_ckpt_epoch'] = 10
        conf['evaluation_epoch'] = 15
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 12
        
        
    def baseline(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['1101']
        conf['train_period'] =  ['20130102', '20130811']
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['model'] = 'baseline_LuongAtt_lstm'
        conf['checkpoint_dir'] = './model/baseline_reg_dropout'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 16
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 32
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
    def baseline_cls(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = './Data/all_data.pkl'
        conf['input_stocks'] = ['1101']
        conf['train_period'] =  ['20130102', '20130711']
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['model'] = 'baseline_LuongAtt_lstm_cls'
        conf['checkpoint_dir'] = './model/baseline_reg_dropout_cls'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 10
        conf['predict_step'] = 5
        conf['batch_size'] = 8
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 32
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
        
    def baseline_2in1(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        conf['train_period'] =  ['20130102', '20140502']
        conf['eval_period'] =  ['20170102', '20180311']
        
        #Model Attributes
        conf['sample_type'] = '2in1'
        conf['model'] = 'baseline_LuongAtt_lstm_2in1'
        conf['checkpoint_dir'] = './model/baseline_reg_dropout_2in1'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 10
        conf['predict_step'] = 5
        conf['batch_size'] = 8
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 32
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
       