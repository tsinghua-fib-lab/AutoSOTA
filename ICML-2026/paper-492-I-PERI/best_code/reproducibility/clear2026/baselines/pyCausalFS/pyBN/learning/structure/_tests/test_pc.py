"""
********
UnitTest
PC
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.structure_learn.path_condition import pc


class PCTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		path = (os.path.join(self.dpath,'lizards.csv'))
		self.data = np.loadtxt(path, dtype='int32',skiprows=1,delimiter=',')
	
	def tearDown(self):
		pass

	def test_pc1_V(self):
		bn = pc(self.data)
		self.assertListEqual(bn.V,
			[0,1,2])

	def test_pc1_E(self):
		bn = pc(self.data)
		self.assertDictEqual(bn.E,
			{0:[1,2],1:[],2:[]})
	
	def test_pc1_F(self):
		bn = pc(self.data)
		self.assertDictEqual(bn.F,
			{0: {'cpt': [], 'parents': [], 'values': [1,2]},
			 1: {'cpt': [], 'parents': [0], 'values': [1,2]},
			 2: {'cpt': [], 'parents': [0], 'values': [1,2]}})
