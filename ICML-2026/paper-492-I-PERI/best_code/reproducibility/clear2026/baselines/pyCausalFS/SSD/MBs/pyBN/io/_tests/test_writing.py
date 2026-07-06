"""
********
UnitTest
Writing
********

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import unittest


def suite():
	suite = unittest.TestLoader().loadTestsFromTestCase(WritingTestCase)
	return suite


class WritingTestCase(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass