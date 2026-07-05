"""
Collection of Graph Algorithms.

Any networkx dependencies should exist here only, but
there are currently a few exceptions which will
eventually be corrected.

"""

import networkx as nx
import numpy as np
from copy import copy

def would_cause_cycle(e, u, v, reverse=False):
	"""
	Test if adding the edge u -> v to the BayesNet
	object would create a DIRECTED (i.e. illegal) cycle.
	"""
	G = nx.DiGraph(e)
	if reverse:
		G.remove_edge(v,u)
	G.add_edge(u,v)
	try:
		nx.find_cycle(G, source=u)
		return True
	except:
		return False



def topsort(edge_dict, root=None):
	"""
	List of nodes in topological sort order from edge dict
	where key = rv and value = list of rv's children
	"""
	queue = []
	if root is not None:
		queue = [root]
	else:
		for rv in edge_dict.keys():
			prior=True
			for p in edge_dict.keys():
				if rv in edge_dict[p]:
					prior=False
			if prior==True:
				queue.append(rv)
	
	visited = []
	while queue:
		vertex = queue.pop(0)
		if vertex not in visited:
			visited.append(vertex)
			for nbr in edge_dict[vertex]:
				queue.append(nbr)
			#queue.extend(edge_dict[vertex]) # add all vertex's children
	return visited

def dfs_postorder(edge_dict, root=None):
	return list(reversed(topsort(edge_dict, root)))

def mst(edge_dict):
	"""
	Calcuate Minimum Spanning Tree
	for a weighted edge dictionary,
	where key = rv, value = another dictionary
	where key = child of rv, value = edge weight
	between rv -> child edge.

	Arguments
	---------
	*w_edge_dict* : a dictionary, where
		key = rv, value = another dictionary, where
		key = child node of "rv", value = integer/float
		edge weight between the "rv" -> "child" edge

	Returns
	-------
	*mst_edge_dict* : a non-weighted edge dict
		The minimum spanning tree from w_edge_dict, stored
		as a dictionary where key = rv, value = list of
		rv's children

	Effects
	-------
	None

	Notes
	-----
	test:
		input:
			g = {0:{1:3,3:2,8:4},1:{7:4},
				2:{3:6,7:2,5:1},3:{4:1},4:{8:8},
				5:{6:8},6:{},7:{2:2},8:{}}
		output:
			g = {0:[3,1,8],1:[7],2:[5],3:[4],4:[],
				5:[6],6:[],7:[2],8:[]}
		
	 
	"""
	mst_G = dict([(rv,[]) for rv in range(len(edge_dict))])
	nrv = len(mst_G)
	reached = [0]
	unreached = range(1,nrv)

	for k in range(nrv):
		source, sink = None, None
		min_cost = np.inf
		for rn in reached:
			e = edge_dict[rn].items()
			e.sort(key = lambda x : x[1])
			for sn, weight in e:
				if sn in unreached and weight < min_cost:
					min_cost = weight
					source = rn
					sink = sn
					break

		if source is None or sink is None:
			break

		# Add e to the minimum spanning tree
		mst_G[source].append(sink)
		#if undirected == True:
		#	mst_G[sink].append(source)

		# Mark newly include node as reached
		unreached.remove(sink)
		reached.append(sink)
	
	return mst_G

def make_chordal(bn, v=None,e=None):
	"""
	This function creates a chordal graph - i.e. one in which there
	are no cycles with more than three nodes.

	Can supply a v_list and e_list for chordal graph of any random graph..

	We start from the moral graph, so if that is already chordal then it
	will return that.

	Algorithm from Cano & Moral 1990 ->
	'Heuristic Algorithms for the Triangulation of Graphs'


	Parameters
	----------
	*v* : a list (optional)
	    A vertex list

	*e* : a list (optional)
	    An edge list


	Returns
	-------
	*G* : a network Digraph object

	Effects
	-------
	None

	Notes
	-----
	Where is this used? Do we need to use networkx?       


	"""
	chordal_E = bn.moralized_edges() # start with moral graph

	# if moral graph is already chordal, no need to alter it
	if not is_chordal(chordal_E):            
		temp_E = copy(chordal_E)
		temp_V = []

		# if v and e is supplied, skip all the rest
		if v and e:
			chordal_E = copy(e)
			temp_E = copy(chordal_E)
			temp_V = copy(v)
		else:
			temp_G = nx.Graph()
			temp_G.add_edges_from(chordal_E)
			degree_dict = temp_G.degree()
			temp_V = sorted(degree_dict, key=degree_dict.get)
		#print temp_V
		for v in temp_V:
			#Add links between the pairs nodes adjacent to Node i
			#Add those links to chordal_E and temp_E
			adj_v = set([n for e in temp_E for n in e if v in e and n!=v])
			for a1 in adj_v:
				for a2 in adj_v:
					if a1!=a2:
						if [a1,a2] not in chordal_E and [a2,a1] not in chordal_E:
							chordal_E.append([a1,a2])
							temp_E.append([a1,a2])
			# remove Node i from temp_V and all its links from temp_E 
			temp_E2 = []
			for edge in temp_E:
				if v not in edge:
					temp_E2.append(edge)
			temp_E = temp_E2

	
	G = nx.Graph()
	G.add_edges_from(chordal_E)
	return G


def is_chordal(edge_list):
	"""
	Check if the graph is chordal/triangulated.

	Parameters
	----------
	*edge_list* : a list of lists (optional)
	The edges to check (if not self.E)

	Returns
	-------
	*nx.is_chordal(G)* : a boolean
	Whether the graph is chordal or not

	Effects
	-------
	None

	Notes
	-----
	Again, do we need networkx for this? Eventually should
	write this check on our own.

	"""
	G = nx.Graph(list(edge_list))
	return nx.is_chordal(G)