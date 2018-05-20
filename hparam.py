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
        conf['feature_size'] = None
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
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        #conf['train_period'] =  ['20130102', '20180811']
        conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = None
        conf['sample_type'] = 'reg'
        conf['model'] = 'baseline_LuongAtt_lstm'
        conf['checkpoint_dir'] = './model/baseline_reg_dropout_5feature_test'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 128
        conf['n_lstm_hidden_units'] = 256
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def baseline_random(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        #conf['train_period'] =  ['20130102', '20150811']
        conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 1
        conf['sample_type'] = 'random'
        conf['model'] = 'baseline_LuongAtt_gru'
        conf['checkpoint_dir'] = '../model/baseline_reg_random_test_f1'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 200
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 32
        conf['n_lstm_hidden_units'] = 32
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_regModel(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        conf['train_period'] =  ['20130102', '20160811']
        #conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = None
        conf['sample_type'] = 'reg'
        conf['model'] = 'baseline_LuongAtt_lstm'
        conf['checkpoint_dir'] = '../model/baseline_reg_test'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 8
        conf['n_lstm_hidden_units'] = 8
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_onlyEnc_GRU(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        #conf['train_period'] =  ['20130102', '20150811']
        conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 44
        conf['sample_type'] = 'random'
        conf['model'] = 'baseline_encReg_gru'
        conf['checkpoint_dir'] = '../model/test_onlyEnc_GRU'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 100
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 128
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 601

    def test_onlyEnc_biderect_gru(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        #conf['train_period'] =  ['20130102', '20150811']
        conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 44
        conf['sample_type'] = 'random'
        conf['model'] = 'baseline_encReg_biderect_gru'
        conf['checkpoint_dir'] = '../model/test_onlyEnc_biderect_gru_orth_init_v2'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 100
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 128
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_onlyEnc_biderect_gru_mstock(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  ['20130102', '20180711']
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 44
        conf['sample_type'] = 'random'
        conf['model'] = 'baseline_encReg_biderect_gru'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_mstocks_step50f128'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 128
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 128
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 24564


    def test_onlyEnc_biderect_gru_allstock(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 10
        conf['sample_type'] = 'reg'
        conf['model'] = 'baseline_encReg_biderect_gru'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_allstocks_step50f128'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 128
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 128
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0


    def test_onlyEnc_biderect_gru_allstock_cls(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 10
        conf['sample_type'] = 'cls'
        conf['model'] = 'baseline_encReg_biderect_gru_cls'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_allstocks_cls_step50f64'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 128
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 64
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_onlyEnc_biderect_gru_nospecialstock_cls(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] =  '/home/dashmoment/workspace/etf_prediction/Data/all_feature_data_Nm[0]_59.pkl'
        conf['meta_file_path'] = '/home/dashmoment/workspace/etf_prediction/Data/all_meta_data_Nm[0]_59.pkl'
        
        #conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm_0_89.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 59
        conf['sample_type'] = 'cls'
        conf['model'] = 'baseline_LuongAtt_lstm_cls'
        #conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_nospecialstock_cls_step50f64'
        conf['checkpoint_dir'] = '/home/dashmoment/tfModel/test_onlyEnc_biderect_gru_nospecialstock_cls_step50f64_all'
        conf['ckpt_name'] = 'baseline_encReg_biderect_gru_cls.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 64
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 100

    def test_onlyEnc_biderect_gru_nospecialstock(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] =  '/home/dashmoment/workspace/etf_prediction/Data/all_feature_data_Nm[0]_59.pkl'
        conf['meta_file_path'] = '/home/dashmoment/workspace/etf_prediction/Data/all_meta_data_Nm[0]_59.pkl'
        
        #conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 59
        conf['sample_type'] = 'reg'
        conf['model'] = 'baseline_encReg_biderect_gru'
        #conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_nospecialstock_step100f128'
        conf['checkpoint_dir'] = '/home/dashmoment/tfModel/test_onlyEnc_biderect_gru_nospecialstock_step100f128'
        
        conf['ckpt_name'] = 'baseline_encReg_biderect_gru.ckpt'
        conf['input_step'] = 100
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 32
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_onlyEnc_biderect_gru_mstock_cls(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] =  '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 59
        conf['sample_type'] = 'cls'
        conf['model'] = 'baseline_encReg_biderect_gru_cls'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_mstocks_cls_f128t50'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 128
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 128
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_onlyEnc_biderect_gru_cls(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        #conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        conf['input_stocks'] = ['0050']
        #conf['train_period'] =  ['20130102', '20170811']
        conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 59
        conf['sample_type'] = 'random_cls'
        conf['model'] = 'baseline_encReg_biderect_gru_cls'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_orth_init_cls_f192t50_lr-2'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 100
        conf['predict_step'] = 5
        conf['batch_size'] = 128
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 192
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 200
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0

    def test_onlyEnc_stacked_gru(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        #conf['train_period'] =  ['20130102', '20150811']
        conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 44
        conf['sample_type'] = 'random'
        conf['model'] = 'baseline_encReg_stacked_gru'
        conf['checkpoint_dir'] = '../model/test_onlyEnc_stacked_gru'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 100
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 64
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0


    def test_regModel_GRU(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_feature_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        conf['train_period'] =  ['20130102', '20160811']
        #conf['train_period'] = None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = None
        conf['sample_type'] = 'reg'
        conf['model'] = 'baseline_LuongAtt_gru'
        conf['checkpoint_dir'] = '../model/baseline_reg_GRU_test_drop08'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 16
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
    def sin_test(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        #conf['src_file_path'] = './Data/all_data.pkl'
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        conf['input_stocks'] = ['0050']
        conf['train_period'] =  ['20130102', '20140811']
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = None
        conf['sample_type'] = 'test'
        conf['model'] = 'baseline_LuongAtt_lstm'
        conf['checkpoint_dir'] = './model/sin_test_Truetrain'
        conf['ckpt_name'] = 'sin_test.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 64
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
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        conf['train_period'] =  ['20130102', '20180711']
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = None
        conf['sample_type'] = 'cls'
        conf['model'] = 'baseline_LuongAtt_lstm_cls'
        conf['checkpoint_dir'] = './model/baseline_reg_dropout_cls'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 64
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 16
        conf['n_lstm_hidden_units'] = 32
        
        #Session Control
        conf['save_ckpt_epoch'] = 101
        conf['evaluation_epoch'] = 500
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 65390
        
        
    def baseline_2in1(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data.pkl'
        #conf['src_file_path'] = './Data/all_data.pkl'
        conf['input_stocks'] = ['0050']
        conf['train_period'] =  ['20130102', '20150702']
        conf['eval_period'] =  ['20170102', '20180311']
        
        #Model Attributes
        conf['sample_type'] = '2in1'
        conf['feature_size'] = None
        conf['model'] = 'baseline_LuongAtt_lstm_2in1'
        conf['checkpoint_dir'] = './model/baseline_reg_dropout_2in1'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 30
        conf['predict_step'] = 5
        conf['batch_size'] = 16
        conf['train_eval_ratio'] = 0.15 
        conf['n_linear_hidden_units'] = 32
        conf['n_lstm_hidden_units'] = 64
        
        #Session Control
        conf['save_ckpt_epoch'] = 11
        conf['evaluation_epoch'] = 50
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
    def baseline_cnn_cls(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = './Data/all_data.pkl'
        conf['input_stocks'] = ['1101', '1102']
        conf['train_period'] =  ['20130102', '20130711']
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = None
        conf['model'] = 'baseline_LuongAtt_lstm_cnn_cls'
        conf['checkpoint_dir'] = './model/baseline_LuongAtt_lstm_cnn_cls'
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
        
    def test_clsModel(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '/home/ubuntu/dataset/etf_prediction/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 10
        conf['sample_type'] = 'cls'
        #conf['model'] = 'baseline_encReg_biderect_gru_cls'
        conf['model'] = 'baseline_LuongAtt_lstm_cls'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_allstocks_cls_step50f64'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 32
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 64
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
    def test_sampleMethod(self):
        
        conf = self.config['common']   
        
        #Data Attributes
        conf['src_file_path'] = '../Data/all_feature_data_Nm[0]_59.pkl'
        #conf['input_stocks'] = ['0050']
        
        conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204']
        #conf['input_stocks'] = ['0050', '0051',  '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '006201', '006203', '006204', '006208','00690', '00692', '00701', '00713']
        #conf['train_period'] =  ['20130102', '20180711']
        conf['train_period'] =  None
        conf['eval_period'] =  ['20170311', '20180402']
        
        #Model Attributes
        conf['feature_size'] = 10
        conf['sample_type'] = 'cls'
        #conf['model'] = 'baseline_encReg_biderect_gru_cls'
        conf['model'] = 'baseline_LuongAtt_lstm_cls'
        conf['checkpoint_dir'] = '/home/ubuntu/model/etf_prediction/test_onlyEnc_biderect_gru_allstocks_cls_step50f64'
        conf['ckpt_name'] = 'baseline_reg.ckpt'
        conf['input_step'] = 50
        conf['predict_step'] = 5
        conf['batch_size'] = 128
        conf['train_eval_ratio'] = 0.2 
        conf['n_linear_hidden_units'] = 64
        conf['n_lstm_hidden_units'] = 64
        
        #Session Control
        conf['save_ckpt_epoch'] = 100
        conf['evaluation_epoch'] = 100
        conf['total_epoch'] = 100000
        conf['current_epoch'] = 0
        
        
       