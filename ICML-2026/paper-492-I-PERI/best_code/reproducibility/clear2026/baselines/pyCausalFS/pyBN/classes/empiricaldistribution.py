"""
************
Empirical
Distribution
Class
************

Create, maintain, and perform operations
over an empirical probability distribution
derived from a dataset. This class is most
useful for speeding up BN structure learning,
where distribution calculations can be cached
and therefore lookups become much faster over
repeated iterations.

References
----------
*** IMPORTANT ***
Daly and Shen,
"Methods to Accelerate the Learning of Bayesian Network Structures."
http://citeseerx.ist.psu.edu/viewdoc/download?
doi=10.1.1.127.7570&rep=rep1&type=pdf
*****************
"""

from __future__ import division

import numpy as np
from copy import copy, deepcopy


class EmpiricalDistribution(object):


	def __init__(self, data, names=None):

		if names is None:
			self.names = range(data.shape[1])
		else:
			assert (len(names) == self.NVAR), 'Passed-in names length must equal number of data columns'
			self.names = names

		self.NROW = data.shape[0]
		self.NVAR = data.shape[1]
		self.bins = [len(np.unique(data[:,n])) for n in range(self.NVAR)]
		

		hist,_ = np.histogramdd(data, bins=self.bins)
		self.counts = hist
		self.joint = (hist / hist.sum()) + 1e-3

		## COMPUTE MARGINAL FOR EACH VARIABLE ##
		#_range = range(self.NVAR)
		#for i,rv in enumerate(self.names):
		#	_axis = copy(_range)
		#	_axis.remove(i)
		#	self.marginal[rv] = np.sum(self.joint,axis=_axis)

		#self.marginal = dict([(rv, np.sum(self.joint,axis=i)) for i,rv in enumerate(self.names)])

		self.cache = {}

	def bayes_counts(self, bn):
		pass

	def idx_map(self, rvs):
		assert (isinstance(args, tuple)), "passed-in rvs must be a tuple"
		idx = [self.names.index(rv) for rv in rvs]
		return tuple(idx)

	def idx(self, rv):
		return self.names.index(rv)

	def axis_tuple(self, rv):
		pass

	def mpd(self, rv):
		"""
		Marginal Probability Distribtuion
		"""
		assert (isinstance(rv, str)), 'passed-in rv must be a string'
		if rv in self.cache:
			mpd = self.cache[rv]
		else:
			_axis = range(self.NVAR)
			_axis.remove(self.idx(rv))
			mpd = np.sum(self.joint, axis=tuple(_axis)) # CORRECT
			self.cache[rv] = mpd

		return mpd

	def jpd(self, rvs):
		assert (isinstance(args, tuple)), "passed-in rvs must be a tuple"
		if args in self.cache:
			jpd = self.cache[rvs]
		else:
			jpd = np.sum(self.joint, axis=self.idx_map(rvs)) # WRONG 
			self.cache[rvs] = jpd
			
		return jpd

	def cpd(self, lhs, rhs):
		assert (isinstance(rhs, tuple)), "passed-in rhs must be a tuple"

		_key = tuple((lhs,rhs))
		if _key in self.cache:
			cpd = self.cache[_key]
		else:
			try:
				tot = lhs + rhs
			except TypeError:
				tot = (lhs,) + rhs

			_numer = self.jpd(tot)
			_denom = self.jpd(rhs)
			cpd = _numer / (_denom + 1e-7)
			self.cache[_key] = cpd
		
		return cpd

	def mi(self, lhs, rhs, cond=None):
		"""
		Calculate mutual information with speedups
		by using caching for repeated lookups of
		joint/conditional distributions.

		"""
		bins = np.amax(data, axis=0) # read levels for each variable
		if len(bins) == 1:
			hist,_ = np.histogramdd(data, bins=(bins)) # frequency counts
			Px = hist/hist.sum()
			MI = -1 * np.sum( Px * np.log( Px ) )
			return round(MI, 4)
			
		if len(bins) == 2:
			hist,_ = np.histogramdd(data, bins=bins[0:2]) # frequency counts

			Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
			Px = np.sum(Pxy, axis = 1) # P(X,Z)
			Py = np.sum(Pxy, axis = 0) # P(Y,Z)	

			PxPy = np.outer(Px,Py)
			Pxy += 1e-7
			PxPy += 1e-7
			MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
			return round(MI,4)
		elif len(bins) > 2 and conditional==True:
			# CHECK FOR > 3 COLUMNS -> concatenate Z into one column
			if len(bins) > 3:
				data = data.astype('str')
				ncols = len(bins)
				for i in range(len(data)):
					data[i,2] = ''.join(data[i,2:ncols])
				data = data.astype('int')[:,0:3]

			bins = np.amax(data,axis=0)
			hist,_ = np.histogramdd(data, bins=bins) # frequency counts

			Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z
			Pz = np.sum(Pxyz, axis = (0,1)) # P(Z)
			Pxz = np.sum(Pxyz, axis = 1) # P(X,Z)
			Pyz = np.sum(Pxyz, axis = 0) # P(Y,Z)	

			Pxy_z = Pxyz / (Pz+1e-7) # P(X,Y | Z) = P(X,Y,Z) / P(Z)
			Px_z = Pxz / (Pz+1e-7) # P(X | Z) = P(X,Z) / P(Z)	
			Py_z = Pyz / (Pz+1e-7) # P(Y | Z) = P(Y,Z) / P(Z)

			Px_y_z = np.empty((Pxy_z.shape)) # P(X|Z)P(Y|Z)
			for i in range(bins[0]):
				for j in range(bins[1]):
					for k in range(bins[2]):
						Px_y_z[i][j][k] = Px_z[i][k]*Py_z[j][k]
			Pxyz += 1e-7
			Pxy_z += 1e-7
			Px_y_z += 1e-7
			MI = np.sum(Pxyz * np.log(Pxy_z / (Px_y_z)))
			
			return round(MI,4)
		elif len(bins) > 2 and conditional == False:
			data = data.astype('str')
			ncols = len(bins)
			for i in range(len(data)):
				data[i,1] = ''.join(data[i,1:ncols])
			data = data.astype('int')[:,0:2]

			hist,_ = np.histogramdd(data, bins=bins[0:2]) # frequency counts

			Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
			Px = np.sum(Pxy, axis = 1) # P(X,Z)
			Py = np.sum(Pxy, axis = 0) # P(Y,Z)	

			PxPy = np.outer(Px,Py)
			Pxy += 1e-7
			PxPy += 1e-7
			MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
			return round(MI,4)











