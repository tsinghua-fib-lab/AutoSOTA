
__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np


def loopy_bp(target=None, evidence=None, max_iter=100):
	"""
	Perform Message Passing (Loopy Belief Propagation) 
	over a cluster graph.

	This is a good candidate for Numba JIT.

	See Koller pg. 397.


	Parameters
	----------


	Returns
	-------


	Notes
	-----
	- Copied from clustergraph class... not tested.
	- Definitely a problem due to normalization (prob vals way too small)
	- Need to check the scope w.r.t. messages.. all clusters should not
	be accumulating rv's in their scope over the course of the algorithm.
	"""
	def collect_beliefs(cgraph):
		cgraph.beliefs = {}
		for cluster in self.V:
			cgraph.V[cluster].collect_beliefs()
			#print('Belief ' , cluster , ' : \n', self.V[cluster].belief.cpt)
			cgraph.beliefs[cluster] = cgraph.V[cluster].belief

	# 1: Moralize the graph
	# 2: Triangluate
	# 3: Build a clique tree using max spanning
	# 4: Propagation of probabilities using message passing

	# creates clique tree and assigns factors, thus satisfying steps 1-3
	cgraph = ClusterGraph(bn)

	edge_visit_dict = dict([(i,0) for i in cgraph.E])

	iteration = 0
	while not cgraph.is_calibrated():
		if iteration == max_iter:
			break
		if iteration % 50 == 0:
			print('Iteration: ' , iteration)
			for cluster in cgraph.V.values():
				cluster.collect_beliefs()
		# select an edge
		e_idx = np.random.randint(0,len(cgraph.E))
		edge_select = cgraph.E[e_idx]
		p_idx = np.random.randint(0,2)
		parent_edge = edge_select[p_idx]
		child_edge = edge_select[np.abs(p_idx-1)]
		print(parent_edge , child_edge)

		# send a message along that edge
		cgraph.V[parent_edge].send_message(cgraph.V[child_edge])

		iteration += 1
	print('Now Collecting Beliefs..')
	collect_beliefs(cgraph)
	#bn.ctree = self