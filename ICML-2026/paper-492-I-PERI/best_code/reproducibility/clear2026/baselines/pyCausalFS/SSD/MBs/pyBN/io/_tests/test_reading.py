"""
********
UnitTest
Reading
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
from pyBN.readwrite.read import read_bn
import os
from os.path import dirname


class ReadingTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn_bif = read_bn(os.path.join(self.dpath,'cancer.bif'))
		self.bn_bn = read_bn(os.path.join(self.dpath,'cmu.bn'))

	def tearDown(self):
		pass

	def test_read_bn_vertices(self):
		self.assertListEqual(self.bn_bn.V, ['Burglary','Earthquake','Alarm','JohnCalls','MaryCalls'])

	def test_read_bif_vertices(self):
		self.assertListEqual(self.bn_bif.V, ['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'])