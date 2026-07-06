"""
**************
UnitTest
Marginal Exact
**************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.readwrite.read import read_bn
from pyBN.inference.marginal_exact import marginal_ve_e



class MarginalExactTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.dpath,'cmu.bn'))
		
	def tearDown(self):
		pass

	def test_marginal_ve_e_prior1(self):
		self.assertListEqual(list(marginal_ve_e(self.bn,'Burglary')),
			self.bn.cpt('Burglary'))

	def test_marginal_ve_e_prior2(self):
		self.assertListEqual(list(marginal_ve_e(self.bn,'Earthquake')),
			self.bn.cpt('Earthquake'))

	#def test_marginal_ve_e_middle(self):
	#	self.assertListEqual(list(marginal_ve_e(self.bn,'Alarm')),
	#		[0.9975,0.0025])

	#def test_marginal_ve_e_leaf1(self):
	#	self.assertListEqual(list(marginal_ve_e(self.bn,'JohnCalls')),
	#		[ 0.88567,  0.11433])

	#def test_marginal_ve_e_prior1_prior_ev(self):
	#	self.assertListEqual(list(marginal_ve_e(self.bn,'Earthquake')),
	#		marginal_ve_e(self.bn,'Earthquake',evidence={'Burglary':'Yes'}))

	#def test_marginal_ve_e_prior2_prior_ev(self):
#		self.assertListEqual(list(marginal_ve_e(self.bn,'Burglary')),
#			marginal_ve_e(self.bn,'Burglary',evidence={'Earthquake':'Yes'}))

	def test_marginal_ve_e_prior_middle_ev(self):
		self.assertListEqual(list(marginal_ve_e(self.bn,'Burglary',
			evidence={'Alarm':'Yes'})),[ 0.62963,  0.37037])

	def test_marginal_ve_e_prior_leaf_ev(self):
		self.assertListEqual(list(marginal_ve_e(self.bn,'Burglary',
			evidence={'JohnCalls':'Yes'})),[ 0.999,  0.001])

	def test_marginal_ve_e_middle_prior_ev(self):
		self.assertListEqual(list(marginal_ve_e(self.bn,'Alarm',
			evidence={'Burglary':'Yes'})),[0.06,0.94])

#	def test_marginal_ve_e_middle_leaf_ev(self):
#		self.assertListEqual(list(marginal_ve_e(self.bn,'Alarm',
#			evidence={'JohnCalls':'Yes'})),[ 0.95769,  0.04231])

	#def test_marginal_ve_e_leaf_prior_ev(self):
	#	self.assertListEqual(list(marginal_ve_e(self.bn,'JohnCalls',
#			evidence={'Burglary':'Yes'})),[ 0.17431,  0.82569])






