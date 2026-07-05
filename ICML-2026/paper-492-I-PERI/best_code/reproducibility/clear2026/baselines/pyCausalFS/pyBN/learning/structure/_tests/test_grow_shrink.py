"""
********
UnitTest
GrowShrink
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.structure_learn.grow_shrink import gs



class GrowShrinkTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		path = (os.path.join(self.dpath,'lizards.csv'))
		self.data = np.loadtxt(path, dtype='int32',skiprows=1,delimiter=',')

	def tearDown(self):
		pass

	def test_gs1_V(self):
		bn = gs(self.data)
		self.assertListEqual(bn.V,
			[0,1,2])

	def test_gs1_E(self):
		bn = gs(self.data)
		self.assertDictEqual(bn.E,
			{0:[1,2],1:[],2:[]})
	
	def test_gs1_F(self):
		bn = gs(self.data)
		self.assertDictEqual(bn.F,
			{0: {'cpt': [], 'parents': [], 'values': [1,2]},
			 1: {'cpt': [], 'parents': [0], 'values': [1,2]},
			 2: {'cpt': [], 'parents': [0], 'values': [1,2]}})

	def test_gs_data(self):
		d = np.loadtxt(os.path.join(self.dpath,'gs_data.txt'),dtype='int32')
		bn = gs(d)
		self.assertDictEqual(bn.E,
			{0: [], 1: [], 2: [0, 1, 4, 3], 3: [], 4: []})





