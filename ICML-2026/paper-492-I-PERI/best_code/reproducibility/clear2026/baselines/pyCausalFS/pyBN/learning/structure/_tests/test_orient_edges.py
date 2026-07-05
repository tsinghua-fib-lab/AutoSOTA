"""
********
UnitTest
OrientEdges
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.structure_learn.orient_edges import orient_edges_gs, orient_edges_pc


class OrientEdgesTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass

	def test_orient_edges_pc(self):
		e = {0:[1,2],1:[0],2:[0]}
		b = {0: [], 1: {2: (0,)}, 2: {1: (0,)}}
		self.assertDictEqual(orient_edges_pc(e,b),
			{0: [1, 2], 1: [], 2: []})

	def test_orient_edges_gs(self):
		e = {0: [1, 2], 1: [0], 2: [0]}
		b = {0: [1, 2], 1: [0], 2: [0]}
		dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		path = (os.path.join(dpath,'lizards.csv'))
		data = np.loadtxt(path, dtype='int32',skiprows=1,delimiter=',')
		self.assertDictEqual(orient_edges_gs(e,b,data,0.05),
			{0: [1, 2], 1: [], 2: []})