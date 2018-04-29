import hparam
config = hparam.configuration()
import numpy as np
import pandas as pd

import pickle
from utility import print_c
import os

class pos_feature_process:

	def __init__(self, file_path = './Data/all_data.pkl'):

		self.file_path = file_path

	def read_file(self):

		print_c("Read pickle data")

		f = open(self.file_path, 'rb')
		_ = pickle.load(f)
		_ = pickle.load(f)
		process_data = pd.DataFrame(pickle.load(f))

		print_c("Finish read pickle data")

		return process_data


	def select_single_stock(self, process_data, stock_number):

		print_c("Process data")

		#select date
		mask = (process_data.columns >= config.start_time) & (process_data.columns < config.end_time)
		select_date = process_data.iloc[:,mask]


		#drop NA
		select_date = select_date.dropna()

		for c in select_date.columns:    
		    for idx in select_date[c].index:
		        select_date[c].loc[idx] = select_date[c].loc[idx].drop(['ID', 'Date','name','trade']).tolist()
		        
		#make a stck_df to np_array 
		a_stock = select_date.loc[stock_number]
		a_stock = np.vstack(np.array(a_stock))

		return a_stock

	def get_single_stock(self, stock_number):

		process_data = self.read_file()
		stock = self.select_single_stock(process_data, stock_number)

#pfp = pos_feature_process()
#stock = pfp.get_single_stock('1101')

#print(np.shape(stock))