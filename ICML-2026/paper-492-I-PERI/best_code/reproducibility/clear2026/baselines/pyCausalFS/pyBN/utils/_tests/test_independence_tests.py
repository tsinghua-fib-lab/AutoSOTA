"""
****************
UnitTest
Constraint Tests
****************

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
import os
from os.path import dirname
import numpy as np

from pyBN.independence.constraint_tests import mi_test

class ConstraintTestsTestCase(unittest.TestCase):

	def setUp(self):
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.data = np.loadtxt(os.path.join(self.dpath,'lizards.csv'),
			delimiter=',',
			dtype='int32',
			skiprows=1)

	def tearDown(self):
		pass

	def test_mi_two_vars_value_a(self):
		self.assertEqual(mi_test(self.data[:,(0,1)]),0.0004)

	def test_mi_two_vars_value_b(self):
		self.assertEqual(mi_test(self.data[:,(0,2)]),0.0014)

	def test_mi_two_vars_symmetry(self):
		self.assertEqual(mi_test(self.data[:,(1,0)]),mi_test(self.data[:,(0,1)]))

	def test_mi_three_vars_value_a(self):
		self.assertEqual(mi_test(self.data),0.0009)

	def test_mi_three_vars_symmetry(self):
		self.assertEqual(mi_test(self.data[:,(0,1,2)]),mi_test(self.data[:,(1,0,2)]))

	def test_mi_random_three(self):
		np.random.seed(3636)
		self.data = np.random.randint(1,10,size=((10000,3)))
		self.assertEqual(mi_test(self.data),0.0211)

	def test_mi_random_four(self):
		np.random.seed(3636)
		self.data = np.random.randint(1,10,size=((10000,4)))
		self.assertEqual(mi_test(self.data),0.0071)











