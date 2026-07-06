"""
********
UnitTest
BayesNet
********

Method	Checks that
assertEqual(a, b)	a == b	 
assertNotEqual(a, b)	a != b	 
assertTrue(x)	bool(x) is True	 
assertFalse(x)	bool(x) is False	 
assertIs(a, b)	a is b
assertIsNot(a, b)	a is not b
assertIsNone(x)	x is None
assertIsNotNone(x)	x is not None
assertIn(a, b)	a in b
assertNotIn(a, b)	a not in b
assertIsInstance(a, b)	isinstance(a, b)
assertNotIsInstance(a, b)	not isinstance(a, b)

assertAlmostEqual(a, b)	round(a-b, 7) == 0	 
assertNotAlmostEqual(a, b)	round(a-b, 7) != 0	 
assertGreater(a, b)	a > b
assertGreaterEqual(a, b)	a >= b
assertLess(a, b)	a < b
assertLessEqual(a, b)	a <= b

assertListEqual(a, b)	lists
assertTupleEqual(a, b)	tuples
assertSetEqual(a, b)	sets or frozensets
assertDictEqual(a, b)	dicts

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest
from pyBN.classes.bayesnet import BayesNet
from pyBN.readwrite.read import read_bn

import os
from os.path import dirname


class BayesNetTestCase(unittest.TestCase):

	def setUp(self):
		self.bn = BayesNet()
		self.dpath = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn_bif = read_bn(os.path.join(self.dpath,'cancer.bif'))
		self.bn_bn = read_bn(os.path.join(self.dpath,'cmu.bn'))

	def tearDown(self):
		pass

	def test_isinstance(self):
		self.assertIsInstance(self.bn,BayesNet)

	def test_V_bif(self):
		self.assertListEqual(self.bn_bif.V,
			['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'])

	def test_E_bif(self):
		self.assertDictEqual(self.bn_bif.E,
			{'Cancer': ['Xray', 'Dyspnoea'],
			 'Dyspnoea': [],
			 'Pollution': ['Cancer'],
			 'Smoker': ['Cancer'],
			 'Xray': []})

	def test_F_bif(self):
		self.assertDictEqual(self.bn_bif.F,
			{'Cancer': {'cpt': [0.03, 0.97, 0.05, 0.95, 0.001, 0.999, 0.02, 0.98],
				  'parents': ['Pollution', 'Smoker'],
				  'values': ['True', 'False']},
			 'Dyspnoea': {'cpt': [0.65, 0.35, 0.3, 0.7],
				  'parents': ['Cancer'],
				  'values': ['True', 'False']},
			 'Pollution': {'cpt': [0.9, 0.1], 'parents': [], 'values': ['low', 'high']},
			 'Smoker': {'cpt': [0.3, 0.7], 'parents': [], 'values': ['True', 'False']},
			 'Xray': {'cpt': [0.9, 0.1, 0.2, 0.8],
				  'parents': ['Cancer'],
				  'values': ['positive', 'negative']}})

	def test_V_bn(self):
		self.assertListEqual(self.bn_bn.V,
			['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'])
	
	def test_E_bn(self):
		self.assertDictEqual(self.bn_bn.E,
			{'Alarm': ['JohnCalls', 'MaryCalls'],
				 'Burglary': ['Alarm'],
				 'Earthquake': ['Alarm'],
				 'JohnCalls': [],
				 'MaryCalls': []})

	def test_F_bn(self):
		self.assertDictEqual(self.bn_bn.F,
			{'Alarm': {'cpt': [0.999, 0.001, 0.71, 0.29, 0.06, 0.94, 0.05, 0.95],
			  'parents': ['Earthquake', 'Burglary'],
			  'values': ['No', 'Yes']},
			 'Burglary': {'cpt': [0.999, 0.001], 'parents': [], 'values': ['No', 'Yes']},
			 'Earthquake': {'cpt': [0.998, 0.002], 'parents': [], 'values': ['No', 'Yes']},
			 'JohnCalls': {'cpt': [0.95, 0.05, 0.1, 0.9],
			  'parents': ['Alarm'],
			  'values': ['No', 'Yes']},
			 'MaryCalls': {'cpt': [0.99, 0.01, 0.3, 0.7],
			  'parents': ['Alarm'],
			  'values': ['No', 'Yes']}})

	def test_nodes(self):
		n = list(self.bn_bn.nodes())
		self.assertListEqual(n,
		['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'])

	def test_cpt(self):
		cpt = list(self.bn_bn.cpt('Alarm'))
		self.assertListEqual(cpt,
			[0.999, 0.001, 0.71, 0.29, 0.06, 0.94, 0.05, 0.95])

	def test_card(self):
		self.assertEqual(self.bn_bn.card('Alarm'),2)

	def test_scope(self):
		self.assertListEqual(self.bn_bn.scope('Alarm'),
			['Alarm', 'Earthquake', 'Burglary'])
	
	def test_parents(self):
		self.assertListEqual(self.bn_bn.parents('Alarm'),
			['Earthquake','Burglary'])

	def test_values(self):
		self.assertListEqual(self.bn_bn.values('Alarm'),['No','Yes'])

	def test_values_idx(self):
		self.assertEqual(self.bn_bn.values('Alarm')[1],'Yes')

if __name__ == '__main__':
	unittest.main(exit=False)




