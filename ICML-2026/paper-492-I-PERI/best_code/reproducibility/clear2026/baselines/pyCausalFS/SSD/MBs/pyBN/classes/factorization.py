"""
*************
Factorization 
Class
*************
"""
from pyBN.classes.factor import Factor

from collections import OrderedDict
from copy import copy,deepcopy
import numpy as np

class Factorization(object):

	def __init__(self, bn, nodes=None):
		"""
		Initialize a Factorization object.

		The *nodes* argument allows you to make a Factorization object
		that includes only a subset of the random variables in the
		passed-in BayesNet object. This is mostly useful for CliqueTree
		algorithms.
		
		"""
		self.bn = bn

		if nodes is not None:
			self._phi = [Factor(bn,rv) for rv in nodes]
		else:
			self._phi = [Factor(bn,rv) for rv in bn.nodes()]

		## MAP-BASED ATTRIBUTES ##
		self.map_factors = OrderedDict()
		self.map_assignment = dict()
		self.map_prob = -1

	def refresh(self):
		"""
		Refresh Factorization attributes.
		"""
		self._phi = [Factor(self.bn,rv) for rv in self.bn.nodes()]
		self.map_factors = OrderedDict()
		self.map_assignment = dict()
		self.map_prob = -1

	def __getitem__(self, idx):
		return self._phi[idx]

	def __iter__(self):
		for phi in self._phi:
			yield phi

	def __len__(self):
		return len(self._phi)

	def __div__(self, rv):
		"""
		Overloads division operator for
		Sum-Product Variable Elimination pass
		over passed-in var.
		"""
		self.sum_product_eliminate_var(rv)
		return self

	def __floordiv__(self, rv):
		"""
		Overloads floor division operator for
		Max-Product Variable Elimination pass
		over passed-in var.
		"""
		self.max_product_eliminate_var(rv)
		return self

	def __sub__(self, rv_val):
		"""
		Overloads subtract operator for
		reducing EVERY relevant factor
		by passed-in rv_val
		"""
		self.map_assignment[rv_val[0]] = rv_val[1] # append
		for phi in self._phi:
			if rv_val[0] in phi.scope:
				phi -= (rv_val[0],rv_val[1]) # reduce by evidence
		return self

	def sum_product_eliminate_var(self, rv):
		relevant_factors = self.relevant_factors(rv)
		irrelevant_factors = self.irrelevant_factors(rv)

		# mutliply all relevant factors
		psi = relevant_factors[0]
		for i in range(1,len(relevant_factors)):
			psi *= relevant_factors[i]

		# Take Sum over psi for rv
		psi /= rv # sumout
		irrelevant_factors.append(psi) # add sum-prod factor back in

		self._phi = irrelevant_factors

	def max_product_eliminate_var(self, rv):
		relevant_factors = self.relevant_factors(rv)
		irrelevant_factors = self.irrelevant_factors(rv)

		# mutliply all relevant factors
		psi = relevant_factors[0]
		for i in range(1,len(relevant_factors)):
			psi *= relevant_factors[i]

		self.map_factors[rv] = deepcopy(psi)
		# Take Max over psi for rv
		psi //= rv # maxout
		irrelevant_factors.append(psi) # add sum-prod factor back in

		self._phi = irrelevant_factors

	def traceback_map(self):
		nodes = self.map_factors.keys()
		idx = len(self.map_factors)
		for rv in reversed(self.map_factors.keys()):
			f = self.map_factors[rv]
			# reduce factor by variables upstream
			for i in range(idx,len(self.map_factors)):
				if nodes[i] in f.scope:
					f -= (nodes[i],self.map_assignment[nodes[i]])
			# choose maximizing assignment conditioned on upstream vars
			self.map_assignment[rv] = self.bn.values(rv)[np.argmax(self.map_factors[rv].cpt)]
			idx-=1
		return self.map_assignment

	def consolidate(self):
		final_phi = self._phi[0]
		for i in range(1,len(self._phi)):
			final_phi *= _phi[i]
		#final_phi.normalize()
		return final_phi

	def relevant_factors(self, rv):
		return [f for f in self._phi if rv in f.scope]

	def irrelevant_factors(self, rv):
		return [f for f in self._phi if rv not in f.scope]










