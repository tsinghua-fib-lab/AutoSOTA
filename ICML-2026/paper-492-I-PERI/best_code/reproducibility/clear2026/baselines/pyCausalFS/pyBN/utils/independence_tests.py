"""
******************************
Conditional Independence Tests 
for Constraint-based Learning
******************************

Implemented Constraint-based Tests
----------------------------------
- mutual information
- Pearson's X^2

I may consider putting this code into its own class structure. The
main benefit I could see from doing this would be the ability to 
cache joint/marginal/conditional probabilities for expedited tests.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
from scipy import stats
from pyBN.utils.data import unique_bins

def are_independent(data, alpha=0.05, method='mi_test'):
	pval = mi_test(data)
	if pval < alpha:
		return True
	else:
		return False

def mutual_information(data, conditional=False):
	#bins = np.amax(data, axis=0)+1 # read levels for each variable
	bins = unique_bins(data)
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



def mi_test(data, test=True):
	"""
	This function performs the mutual information (cross entropy)-based
	CONDITIONAL independence test. Because it is conditional, it requires
	at LEAST three columns. For the marginal independence test, use 
	"mi_test_marginal".

	We use the maximum likelihood estimators as probabilities. The
	mutual information value is computed, then the 
	chi square test is used, with degrees of freedom equal to 
	(|X|-1)*(|Y|-1)*Pi_z\inZ(|z|).

	This function works on datasets that contain MORE than three
	columns by concatenating the extra columns into one. For that
	reason, it is a little slower in that case.

	For two variables only:

	This function performs mutual information (cross entropy)-based
	MARGINAL independence test. Because it is marginal, it requires
	EXACTLY TWO columns. For the conditional independence test, use
	"mi_test_conditional".

	This is the same as calculated the KL Divergence, i.e.
	I(X,Y) = Sigma p(x,y)* log p(x,y) *( p(x)/p(y) )

	NOTE: pval < 0.05 means DEPENDENCE, pval > 0.05 means INDEPENDENCE.
	In other words, the pval represent the probability this relationship
	could have happened at random or by chance. if the pval is very small,
	it means the two variables are likely dependent on one another.

	Steps:
		- Calculate the marginal/conditional probabilities
		- Compute the Mutual Information value
		- Calculate chi2 statistic = 2*N*MI
		- Compute the degrees of freedom
		- Compute the chi square p-value

	Arguments
	----------
	*data* : a nested numpy array
		The data from which to learn - must have at least three
		variables. All conditioned variables (i.e. Z) are compressed
		into one variable.

	Returns
	-------
	*p_val* : a float
		The pvalue from the chi2 and ddof

	Effects
	-------
	None

	Notes
	-----
	- Doesn't currently work with strings... 
	- Should generalize to let data be a Pandas DataFrame --> would
	encourage external use.

	"""
	
	#bins = np.amax(data, axis=0)+1 # read levels for each variable
	bins = unique_bins(data)
	if len(bins)==2:
		hist,_ = np.histogramdd(data, bins=bins[0:2]) # frequency counts

		#Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
		Pxy = hist / data.shape[0]
		Px = np.sum(Pxy, axis = 1) # P(X,Z)
		Py = np.sum(Pxy, axis = 0) # P(Y,Z)	

		PxPy = np.outer(Px,Py)
		Pxy += 1e-7
		PxPy += 1e-7
		MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
		if not test:
			return round(MI,4)
		else:
			chi2_statistic = 2*len(data)*MI
			ddof = (bins[0] - 1) * (bins[1] - 1)
			p_val = 2*stats.chi2.pdf(chi2_statistic, ddof)
			return round(p_val,4)
	else:
		# CHECK FOR > 3 COLUMNS -> concatenate Z into one column
		if len(bins) > 3:
			data = data.astype('str')
			ncols = len(bins)
			for i in range(len(data)):
				data[i,2] = ''.join(data[i,2:ncols])
			data = data.astype('int')[:,0:3]

		#bins = np.amax(data,axis=0)
		bins = unique_bins(data)
		hist,_ = np.histogramdd(data, bins=bins) # frequency counts

		#Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z
		Pxyz = hist / data.shape[0]
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
		if not test:
			return round(MI,4)
		else:
			chi2_statistic = 2*len(data)*MI
			ddof = (bins[0] - 1) * (bins[1] - 1) * bins[2]
			p_val = 2*stats.chi2.pdf(chi2_statistic, ddof) # 2* for one tail
			return round(p_val,4)

def entropy(data):
	"""
	In the context of structure learning, and more specifically
	in constraint-based algorithms which rely on the mutual information
	test for conditional independence, it has been proven that the variable
	X in a set which MAXIMIZES mutual information is also the variable which
	MINIMIZES entropy. This fact can be used to reduce the computational
	requirements of tests based on the following relationship:

		Entropy is related to marginal mutual information as follows:
			MI(X;Y) = H(X) - H(X|Y)

		Entropy is related to conditional mutual information as follows:
			MI(X;Y|Z) = H(X|Z) - H(X|Y,Z)

		For one varibale, H(X) is equal to the following:
			-1 * sum of p(x) * log(p(x))

		For two variables H(X|Y) is equal to the following:
			sum over x,y of p(x,y)*log(p(y)/p(x,y))
		
		For three variables, H(X|Y,Z) is equal to the following:
			-1 * sum of p(x,y,z) * log(p(x|y,z)),
				where p(x|y,z) = p(x,y,z)/p(y)*p(z)
	Arguments
	----------
	*data* : a nested numpy array
		The data from which to learn - must have at least three
		variables. All conditioned variables (i.e. Z) are compressed
		into one variable.

	Returns
	-------
	*H* : entropy value

	"""
	try:
		cols = data.shape[1]
	except IndexError:
		cols = 1

	#bins = np.amax(data,axis=0)
	bins = unique_bins(data)

	if cols == 1:
		hist,_ = np.histogramdd(data, bins=(bins)) # frequency counts
		Px = hist/hist.sum()
		H = -1 * np.sum( Px * np.log( Px ) )

	elif cols == 2: # two variables -> assume X then Y
		hist,_ = np.histogramdd(data, bins=bins[0:2]) # frequency counts

		Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
		Py = np.sum(Pxy, axis = 0) # P(Y)	
		Py += 1e-7
		Pxy += 1e-7
		H = np.sum( Pxy * np.log( Py / Pxy ) )

	else:
		# CHECK FOR > 3 COLUMNS -> concatenate Z into one column
		if cols  > 3:
			data = data.astype('str')
			ncols = len(bins)
			for i in range(len(data)):
				data[i,2] = ''.join(data[i,2:ncols])
			data = data.astype('int')[:,0:3]

		bins = np.amax(data,axis=0)
		hist,_ = np.histogramdd(data, bins=bins) # frequency counts

		Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z
		Pyz = np.sum(Pxyz, axis=0)

		Pxyz += 1e-7 # for log -inf
		Pyz += 1e-7
		H = -1 * np.sum( Pxyz * np.log( Pxyz ) ) + np.sum( Pyz * np.log( Pyz ) ) 

	return round(H,4)

def mi_from_en(data):
	"""
	Calculate Mutual Information based on entropy alone. This
	function isn't really faster than calculating MI from
	"mi_test" due to overhead in the histogram/binning from
	numpy, but it's mostly here for entropy validation or
	if "mi_test" breaks. BUT, calling "entropy" once is much
	faster than calling "mi_test" once, so that is where the
	speedup occurs.

	This has been validated with both 2, 3 and 4 variables.

	Entropy is related to marginal mutual information as follows:
			MI(X;Y) = H(X) - H(X|Y)
	Entropy is related to conditional mutual information as follows:
			MI(X;Y|Z) = H(X|Z) - H(X|Y,Z)
	"""
	ncols = data.shape[1]

	if ncols == 1:
		print("Need at least 2 columns")

	elif ncols==2:
		MI = entropy(data[:,0]) - entropy(data)

	elif ncols==3:
		MI = entropy(data[:,(0,2)]) - entropy(data)

	elif ncols>3:
		# join extra columns
		data = data.astype('str')
		ncols = data.shape[1]
		for i in range(len(data)):
			data[i,2] = ''.join(data[i,2:ncols])
		data = data.astype('int')[:,0:3]

		MI = entropy(data[:,(0,2)]) - entropy(data)


	return round(MI,4)


def chi2_test(data):
	"""
	Test null hypothesis that P(X,Y,Z) = P(Z)P(X|Z)P(Y|Z)
	versus empirically observed P(X,Y,Z) in the data using
	the traditional chisquare test based on observed versus
	expected frequency bins.

	Steps
		- Calculate P(XYZ) empirically and expected
		- Compute ddof
		- Perfom one-way chisquare

	Arguments
	---------
	*data* : a nested numpy array
		The data from which to learn - must have at least three
		variables. All conditioned variables (i.e. Z) are compressed
		into one variable.

	Returns
	-------
	*chi2_statistic* : a float
		Chisquare statistic
	*p_val* : a float
		The pvalue from the chi2 and ddof

	Effects
	-------
	None

	Notes
	-----
	- Assuming for now that |Z| = 1... generalize later
	- Should generalize to let data be a Pandas DataFrame --> would
	encourage external use.

	"""
	# compress extra Z variables at the start.. not implemented yet
	#bins = np.amax(data, axis=0)+1
	bins = unique_bins(data)
	hist,_ = np.histogramdd(data,bins=bins)

	Pxyz = hist / hist.sum()# joint probability distribution over X,Y,Z

	Pz = np.sum(Pxyz, axis = (0,1)) # P(Z)
	Pxz = np.sum(Pxyz, axis = 1) # P(X,Z)
	Pyz = np.sum(Pxyz, axis = 0) # P(Y,Z)

	Px_z = Pxz / (Pz+1e-7) # P(X | Z) = P(X,Z) / P(Z)	
	Py_z = Pyz / (Pz+1e-7) # P(Y | Z) = P(Y,Z) / P(Z)

	observed_dist = Pxyz # Empirical distribution
	#Not correct right now -> Pz is wrong dimension
	Px_y_z = np.empty((Pxy_z.shape)) # P(Z)P(X|Z)P(Y|Z)
	for i in range(bins[0]):
		for j in range(bins[1]):
			for k in range(bins[2]):
				Px_y_z[i][j][k] = Px_z[i][k]*Py_z[j][k]
	Px_y_z *= Pz

	observed = observed_dist.flatten() * len(data)
	expected = expected_dist.flatten() * len(data)

	ddof = (bins[0] - 1) * (bins[1]- 1) * bins[2]
	chi2_statistic, p_val = stats.chisquare(observed,expected, ddof=ddof)

	return chi2_statistic, p_val





