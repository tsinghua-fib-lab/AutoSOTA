"""
********
UnitTest
Map Exact
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.readwrite.read import read_bn
from pyBN.inference.map_exact import map_ve_e

class MapExactTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.dpath,'cmu.bn'))

	def tearDown(self):
		pass

	#def test_map_noevidence(self):
		#p = list(map_ve_e(self.bn,target='Alarm'))
		#self.assertListEqual(p,['No',0.9367])

	#def test_map_prior(self):
		#p = list(map_ve_e(self.bn,target='Burglary'))
		#self.assertListEqual(p,['No',0.9367])

	#def test_map_leaf(self):
		#p = list(map_ve_e(self.bn,target='JohnCalls'))
		#self.assertlistEqual(p,['No',0.9367])
		