"""
Max-Min Parents and Children Algorithm
for determining the Parent-Children set
for all nodes. This returns an unoriented
graph that then must be oriented using
some set of rules or other orientation
algorithm.

Using a version of hill climbing to orient
the result of MMPC results in the MMHC -
Max-Min Hill Climbing algorithm. In other
words, MMPC can be viewed as a subroutine
of MMHC.

References
----------
[1] Tsamardinos, et al. "The max-min hill-climbing Bayesian
network structure learning algorithm."

"""


def mmpc(data, alpha=0.05):
	"""
	Max-Min Parents and Children Algorithm
	for determining the Parent-Children set
	for all nodes. This returns an unoriented
	graph that then must be oriented using
	some set of rules or other orientation
	algorithm.

	Arguments
	---------
	*data* : a numpy ndarray

	*alpha* : a float
		Probability of Type II Error for
		independence tests

	Returns
	-------
	*CPC_dict* : a dictionary, where
		key = rv, value = list of rv's children AND parents

	Notes
	-----
	"""

	nrow = data.shape[0]
	ncol = data.shape[1]
	
	nodes = range(ncol)

	CPC_dict = dict([(n,[]) for n in nodes]) # rv -> children of rvs
	value_dict = dict([(n, np.unique(data[:,i])) for i,n in enumerate(nodes)])

	# LEARN PARENT-CHILD SET FOR EACH NODE
	for T in nodes:
		
		# GROW PHASE
		CPC = []
		changed=True
		while changed:
			changed=False
			# MAX-MIN HEURISTIC
			min_assoc = 1e9
			min_node = None
			for x in nodes:
				if x!=T:
					cols = (x,T) + tuple(CPC)
					pval = mi_test(data[:,cols], test=True)
				if pval < min_assoc:
					min_assoc = pval
					min_node = x
			# only add node if it's small but still dependent
			if min_assoc > alpha:
				CPC.append(min_node)
				changed=True

		# SHRINK PHASE
		for X in CPC:
			cpc = [c for c in CPC if X!=c]
			for i in len(cpc):
				for S in itertools.combinations(cpc,i):
					cols = (T,X) + S
					pval = mi_test(data[:,cols])
					# if I(X,T | S) = TRUE
					if pval_xy_z > alpha:
						CPC.remove(X)

		# ADD CPC TO CPC_dict
		CPC_dict[T] = CPC

	# REMOVE FALSE POSITIVES
	for T in nodes:
		for X in CPC_dict[T]:
			# If X is in CPC[T] but T is not in CPC[X], remove X.
			if T not in CPC_dict[X]:
				CPC_dict[T].remove(X)

	return CPC_dict
























