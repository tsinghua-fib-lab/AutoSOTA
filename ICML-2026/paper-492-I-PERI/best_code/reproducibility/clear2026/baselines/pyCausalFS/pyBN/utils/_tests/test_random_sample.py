"""
********
UnitTest
Misc
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.utils.random_sample import random_sample
from pyBN.readwrite.read import read_bn


class RandomSampleTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.dpath,'cancer.bif'))

	def tearDown(self):
		pass

	def test_random_sample(self):
		np.random.seed(3636)
		sample = random_sample(self.bn,5)
		self.assertListEqual(list(sample.ravel()),
			[0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 
			1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])