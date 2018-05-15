import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np

import data_process_list as dp
import model_zoo as mz


from utility import tf_utility as ut


class regression_score():
    
    def __init__(self, config, stock, mean, std):
        
        self.config = config
        self.stock = stock
        self.mean = mean
        self.std = std
        self.weighted_array = [0.1,0.15,0.2,0.25,0.3]
    
    def regression_score(self):
    
        c = self.config
        tv_gen = dp.train_validation_generaotr()
        _, evalSet , *_ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'], c['input_stocks'],
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        metafile = c['meta_file_path'])
          
        if c['feature_size'] == None: c['feature_size'] = evalSet.shape[-1]
        c['batch_size'] = len(evalSet) 
        
        train, label = np.split(evalSet, [c['input_step']], axis=1)    
        label_reg = label[:,:,3]
       
        tf.reset_default_graph()  
        x = tf.placeholder(tf.float32, [None, c['input_step'],c['feature_size']]) 
        y = tf.placeholder(tf.float32, [None, c['predict_step']]) 
        predict_reg = mz.model_zoo(c, x, y, dropout = 1.0, is_train=False).decoder_output
        
        predict_reg_resore = predict_reg*self.std + self.mean
        label_reg_restore = label_reg*self.std + self.mean
        
        abs_loss = tf.abs(predict_reg_resore-label_reg_restore)
        score_ref = ((label_reg_restore -abs_loss)/label_reg_restore)*0.5
        w_score = self.weighted_array*score_ref
        mean_score = tf.reduce_mean(tf.reduce_sum(w_score, axis=1))
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
                    
            if ut.load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
           
            predict = sess.run(predict_reg, feed_dict={x:train, y:label_reg})
            loss, plain_scores, w_scores , mean_score_reg = sess.run([abs_loss,  score_ref, w_score, mean_score], feed_dict={x:train, y:label_reg})
            
            sess.close()
            
        return mean_score_reg, predict, label_reg

class classification_score():
    
    def __init__(self, config, stock):
        
        self.config = config
        self.stock = stock
        self.weighted_array = [0.1,0.15,0.2,0.25,0.3]
    
    def classification_score(self):
                    
        tf.reset_default_graph()              
        c = self.config
        
        tv_gen = dp.train_validation_generaotr()
        _, evalSet , *_ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'], c['input_stocks'],
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        metafile = c['meta_file_path'])
        
        if c['feature_size'] == None: c['feature_size'] = evalSet.shape[-1]
        c['batch_size'] = len(evalSet) 
        train, label = np.split(evalSet, [c['input_step']], axis=1) 
        #train = np.reshape(train[:,:,:c['feature_size']], (c['batch_size'], c['input_step'], -1))
        label_cls = label[:,:,-3:]   
        
        x = tf.placeholder(tf.float32, [None, c['input_step'],c['feature_size']]) 
        y = tf.placeholder(tf.float32, [None, c['predict_step'], 3]) 
        
        logits = mz.model_zoo(c, x, y, dropout = 1.0, is_train = False).decoder_output
        softmax_out = tf.nn.softmax(logits)
        predict_cls = tf.argmax(tf.nn.softmax(logits), axis=-1)
        ground_truth = tf.argmax(y, axis=-1)
       
        plane_result = tf.cast(tf.equal(predict_cls, ground_truth), tf.float32)
        weighted_result = plane_result*0.5*self.weighted_array
        score_tf = tf.reduce_sum(weighted_result, axis=1)
        mean_score = tf.reduce_mean(score_tf)
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
                    
            if ut.load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
            loss_s, softmax_out_s, predict_s = sess.run([logits, softmax_out,predict_cls], feed_dict={x:train, y:label_cls})
            result_cls, score, w_score, mean_score_cls = sess.run([plane_result, score_tf, weighted_result, mean_score], feed_dict={x:train, y:label_cls})
            
        return mean_score_cls, predict_s


class regression2Cls_score():
    
    def __init__(self, config, stock, mean, std):
        
        self.config = config
        self.stock = stock
        self.mean = mean
        self.std = std
        self.weighted_array = [0.1,0.15,0.2,0.25,0.3]
        
    def generate_cls_from_reg(self, concat_price):
        
        cls_res_raw = []
        for i in range(1,len(concat_price[0])):
            cls_res_raw.append(concat_price[:,i] - concat_price[:, i-1])
        cls_res_raw = np.transpose(np.vstack(cls_res_raw), (1,0))
            
        def mapfuc(x):
                
            if x < 0: return 0
            elif x > 0: return 2
            else: return 1
             
        cls_res = np.zeros(np.shape(cls_res_raw))
        for i in range(len(cls_res_raw)):
            for j in range(len(cls_res_raw[0])):
                cls_res[i,j] = mapfuc(cls_res_raw[i,j])
                
        return cls_res_raw, cls_res

    def regression2Cls_score(self):
    
        c = self.config
        #c['checkpoint_dir'] = './model/test_onlyEnc_biderect_gru_orth_init'
        
        tv_gen = dp.train_validation_generaotr()
        _, evalSet , *_ = tv_gen.generate_train_val_set_mStock(
                                                        c['src_file_path'], c['input_stocks'],
                                                        c['input_step'], c['predict_step'], c['train_eval_ratio'], 
                                                        metafile = c['meta_file_path'])
        
        if c['feature_size'] == None: c['feature_size'] = evalSet.shape[-1]
        c['batch_size'] = len(evalSet)
        train, label = np.split(evalSet, [c['input_step']], axis=1) 
        label_reg = label[:,:,3]
       
        tf.reset_default_graph()  
        x = tf.placeholder(tf.float32, [None, c['input_step'],c['feature_size']]) 
        y = tf.placeholder(tf.float32, [None, c['predict_step']]) 
        predict_reg = mz.model_zoo(c, x, y, dropout = 1.0, is_train=False).decoder_output
        
        predict_reg_resore = predict_reg*self.std + self.mean
        label_reg_restore = y*self.std + self.mean
    
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
                    
            if ut.load_ckpt(saver, sess, c['checkpoint_dir'],  c['ckpt_name']):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
           
            predict_s, label_reg_s = sess.run([predict_reg_resore, label_reg_restore], feed_dict={x:train, y:label_reg})
            
            label_cls = np.argmax(label[:,:,-3:], axis=-1)
            last_price = np.reshape(label_reg_s[:,-1], (-1,1))
            concat_price = np.concatenate((last_price, predict_s), axis = 1)
            
            _, cls_res = self.generate_cls_from_reg(concat_price)
            
            sess.close()
            
        match_matrix = np.equal(cls_res, label_cls)
        match_matrix = match_matrix*0.5*self.weighted_array
        score = np.mean(np.sum(match_matrix, axis=1))
            
        return score, match_matrix, cls_res, label_cls, label_reg,  


