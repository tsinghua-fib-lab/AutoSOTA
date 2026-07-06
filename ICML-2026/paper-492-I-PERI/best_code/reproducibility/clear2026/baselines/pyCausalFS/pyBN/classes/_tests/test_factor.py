"""
********
UnitTest
Factor
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
import os
from os.path import dirname
import numpy as np

from pyBN.classes.factor import Factor
from pyBN.readwrite.read import read_bn


class FactorTestCase(unittest.TestCase):

	def setUp(self):
		self.data_path = os.path.join(dirname(dirname(dirname(dirname(__file__)))),'data')	
		self.bn = read_bn(os.path.join(self.data_path,'cmu.bn'))
		self.f = Factor(self.bn, 'Alarm')
	def tearDown(self):
		pass

	# Factor Creation Tests
	def test_factor_init(self):
		self.assertIsInstance(self.f,Factor)

	def test_factor_bn(self):
		self.assertListEqual(self.f.bn.V,
			['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'])

	def test_factor_var(self):
		self.assertEqual(self.f.var, 'Alarm')
	
	def test_factor_scope(self):
		self.assertListEqual(self.f.scope,['Alarm','Earthquake','Burglary'])
	
	def test_factor_card(self):
		self.assertDictEqual(self.f.card,
			{'Alarm':2, 'Burglary':2, 'Earthquake':2})
	
	def test_factor_stride(self):
		self.assertDictEqual(self.f.stride,
			{'Alarm':1, 'Burglary':4, 'Earthquake':2})
	
	def test_factor_cpt(self):
		self.assertListEqual(list(self.f.cpt),
			[ 0.999,  0.001,  0.71 ,  0.29 ,  0.06 ,  0.94 ,  0.05 ,  0.95 ])
	
	# Factor Operations Tests
	def test_multiply_factor(self):
		f1 = Factor(self.bn,'Alarm')
		f2 = Factor(self.bn,'Burglary')
		f1.multiply_factor(f2)

		f3 = Factor(self.bn,'Burglary')
		f4 = Factor(self.bn,'Alarm')
		f3.multiply_factor(f4)

		self.assertListEqual(list(f1.cpt),list(f3.cpt))
	
	def test_sumover_var(self):
		self.f.sumover_var('Burglary')
		self.assertListEqual(list(self.f.cpt),[0.5,0.5])

	def test_sumout_var_list(self):
		f = Factor(self.bn,'Alarm')
		f.sumout_var_list(['Burglary','Earthquake'])
		self.assertListEqual(f.scope,['Alarm'])
		self.assertDictEqual(f.stride,{'Alarm':1})
		self.assertListEqual(list(f.cpt),[0.45475,0.54525])

	def test_sumout_var(self):
		f = Factor(self.bn,'Alarm')
		f.sumout_var('Earthquake')
		self.assertListEqual(f.scope,['Alarm','Burglary'])
		self.assertDictEqual(f.stride,
			{'Alarm':1,'Burglary':2})
		self.assertListEqual(list(f.cpt),
			[ 0.8545,  0.1455,  0.055 ,  0.945 ])

	def test_maxout_var(self):
		"""
		I KNOW THIS IS CORRECT.
		"""
		f = Factor(self.bn,'Alarm')
		f.maxout_var('Burglary')
		self.assertListEqual(list(f.cpt),
			[ 0.999,  0.94 ,  0.71 ,  0.95 ])
		self.assertListEqual(f.scope,['Alarm','Earthquake'])
		self.assertDictEqual(f.stride,
			{'Alarm':1,'Earthquake':2})

	def test_reduce_factor_by_list(self):
		f = Factor(self.bn, 'Alarm')
		f.reduce_factor_by_list([['Burglary','Yes'],['Earthquake','Yes']])
		self.assertListEqual(list(f.cpt),[0.05,0.95])
		self.assertListEqual(f.scope,['Alarm'])
		self.assertDictEqual(f.stride,{'Alarm':1})

	def test_reduce_factor(self):
		f = Factor(self.bn, 'Alarm')
		f.reduce_factor('Burglary','Yes')
		self.assertListEqual(list(f.cpt),
			[ 0.06,  0.94,  0.05,  0.95])

	def test_to_log(self):
		f = Factor(self.bn,'Earthquake')
		f.to_log()
		self.assertEqual(round(sum(f.cpt),4),-6.2166)

	def test_from_log(self):
		f = Factor(self.bn, 'Earthquake')
		f.to_log()
		f.from_log()
		self.assertListEqual(list(f.cpt),[0.998,0.002])

	def test_normalize(self):
		self.f.cpt[0]=20
		self.f.cpt[1]=20
		self.f.cpt[4]=0.94
		self.f.cpt[7]=0.15
		self.f.normalize()
		self.assertListEqual(list(self.f.cpt),
			[0.500,0.500,0.710,0.290,0.5,0.5,0.25,0.75])
    
if __name__ == '__main__':
	unittest.main(exit=False)









