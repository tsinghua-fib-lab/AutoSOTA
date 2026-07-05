"""
**************************
Discretize Continuous Data
**************************

Since pyBN only handles Discrete Bayesian Networks,
and therefore only handles discrete data, it is
important to have effective functions for 
discretizing continuous data. This code aims to
meet that goal.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
from copy import copy

def discretize(data, cols=None, bins=None):
	"""
	Discretize the passed-in dataset. These
	functions will rely on numpy and scipy
	for speed and accuracy... no need to
	reinvent the wheel here. Therefore, pyBN's
	discretization methods are basically just
	wrappers for existing methods.

	The bin number defaults to FIVE (5) for
	all columns if not passed in.

	Arguments
	---------
	*data* : a nested numpy array

	*cols* : a list of integers (optional)
		Which columns to discretize .. defaults
		to ALL columns

	*bins* : a list of integers (optional)
		The number of bins into which each column
		array will be split .. defaults to 5 for
		all columns

	Returns
	-------
	*data* : a discretized copy of original data

	Effects
	-------
	None

	Notes
	-----
	- Should probably add more methods of discretization
		based on mean/median/mode, etc.
	"""
	if bins is not None:
		assert (isinstance(bins, list)), 'bins argument must be a list'
	else:
		try:
			bins = [5]*data.shape[1]
		except ValueError:
			bins = [5]
	
	if cols is not None:
		assert (isinstance(cols,list)), 'cols argument must be a list'
	else:	
		try:
			cols = range(data.shape[1])
		except ValueError:
			cols = [0]

	data = copy(data)

	minmax = zip(np.amin(data,axis=0),np.amax(data,axis=0))
	for c in cols:
		# get min and max of each column
		_min, _max = minmax[c]
		# create the bins from np.linspace
		_bins = np.linspace(_min,_max,bins[c])
		# discretize with np.digitize(col,bins)
		data[:,c] = np.digitize(data[:,c],_bins)

	return np.array(data,dtype=np.int32,copy=False)










