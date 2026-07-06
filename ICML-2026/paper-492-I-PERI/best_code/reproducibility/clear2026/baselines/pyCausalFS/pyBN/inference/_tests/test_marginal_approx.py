"""
***************
UnitTest
Marginal Approx
***************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.inference.marginal_approx import marginal_fs_a, marginal_lws_a, marginal_gs_a
from pyBN.readwrite.read import read_bn

class MarginalApproxTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.dpath,'cmu.bn'))

	def tearDown(self):
		pass

	def test_forward_sample(self):
		np.random.seed(3636)
		self.assertDictEqual(marginal_fs_a(self.bn,n=1000),
			{'Alarm': {'No': 0.996, 'Yes': 0.004},
			 'Burglary': {'No': 0.998, 'Yes': 0.002},
			 'Earthquake': {'No': 1.0, 'Yes': 0.0},
			 'JohnCalls': {'No': 0.946, 'Yes': 0.054},
			 'MaryCalls': {'No': 0.99, 'Yes': 0.01}}
			 )

	def test_likelihood_weighted_sample(self):
		"""
		THIS HAS BEEN VALIDATED AGAINST BNLEARN
		"""
		np.random.seed(3636)
		self.assertDictEqual(marginal_lws_a(self.bn,evidence={'Burglary':'Yes'}),
			{'Alarm': {'No': 0.058, 'Yes': 0.942},
			 'Burglary': {'No': 0.0, 'Yes': 1.0},
			 'Earthquake': {'No': 0.997, 'Yes': 0.003},
			 'JohnCalls': {'No': 0.136, 'Yes': 0.864},
			 'MaryCalls': {'No': 0.343, 'Yes': 0.657}})

	def test_lws_allevidence(self):
		self.assertDictEqual(marginal_lws_a(self.bn,evidence={'Burglary':'Yes','Alarm':'Yes',
			'Earthquake':'Yes','JohnCalls':'Yes','MaryCalls':'Yes'}),
			{'Alarm': {'No': 0.0, 'Yes': 1.0},
			 'Burglary': {'No': 0.0, 'Yes': 1.0},
			 'Earthquake': {'No': 0.0, 'Yes': 1.0},
			 'JohnCalls': {'No': 0.0, 'Yes': 1.0},
			 'MaryCalls': {'No': 0.0, 'Yes': 1.0}})

	def test_gibbs(self):
		np.random.seed(3636)
		self.assertDictEqual(marginal_gs_a(self.bn,n=1000,burn=200),
			{'Alarm': {'No': 0.9988, 'Yes': 0.0},
				 'Burglary': {'No': 0.9988, 'Yes': 0.0},
				 'Earthquake': {'No': 0.9975, 'Yes': 0.0013},
				 'JohnCalls': {'No': 0.9313, 'Yes': 0.0675},
				 'MaryCalls': {'No': 0.9838, 'Yes': 0.015}})










