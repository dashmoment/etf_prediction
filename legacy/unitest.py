import unittest
import numpy as np
import pos_feature_process

class Test_pfp_methods(unittest.TestCase):

	def __init__(self, *args, **kwargs):

		super(Test_pfp_methods, self).__init__(*args, **kwargs)
		self.pfp = pos_feature_process.pos_feature_process()
	
	def test_read_file(self):

		f = self.pfp.read_file()
		print('load file shape: {}'.format(np.shape(f)))
		self.assertTrue(len(np.shape(f)) > 0)

		s = self.pfp.select_single_stock(f, '1101')
		print('a stock shape: {}'.format(np.shape(s)))
		Test_pfp_methods.shape = np.shape(s)
		self.assertTrue(len(np.shape(s)) > 0)

	def test_get_single_stock(self):

		s = self.pfp.get_single_stock('1101')
		print(s, self.shape)
		self.assertEqual(np.shape(s), Test_pfp_methods.shape)



if __name__ == '__main__':

	alltests = unittest.TestSuite()
	alltests.addTest(unittest.makeSuite(Test_pfp_methods))
	result = unittest.TextTestRunner(verbosity=2).run(alltests)
