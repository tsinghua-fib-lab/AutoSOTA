Documentation of treehfd
========================


**treehfd** is a Python module to compute the Hoeffding functional decomposition
of XGBoost models (Chen and Guestrin, 2016) with dependent input variables, using
the TreeHFD algorithm. 
This decomposition breaks down XGBoost models into a sum of functions of one or 
two variables (respectively main effects and interactions), which are intrinsically
explainable, while preserving the accuracy of the initial black-box model.

**treehfd** currently supports XGBoost models for regression and binary classification
with numeric input variables, and categorical variables should be one-hot encoded.


Scientific Background
---------------------

The TreeHFD algorithm is introduced in the following article: 
`Bénard, C. (2025). Tree Ensemble Explainability through the Hoeffding Functional Decomposition and TreeHFD Algorithm.
In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), in press. <https://openreview.net/pdf?id=dRLWcpBQxS>`_

More precisely, the Hoeffding decomposition was introduced by Hoeffding (1948)
for independent input variables. More recently, Stone (1994) and Hooker (2007)
extended the decomposition to the case of dependent input variables, where uniqueness
of the decomposition is enforced by hierarchical orthogonality constraints. The estimation
from a data sample in the dependent case is a notoriously difficult problem, and 
therefore, the Hoeffding decomposition has long remain an abstract theoretical tool.
TreeHFD computes the Hoeffding decomposition of tree ensembles, based on a 
discretization of the hierarchical orthogonality constraints over the
Cartesian tree partitions. Such decomposition is proved to be piecewise constant 
over these partitions, and the values in each cell for all components are given 
by solving a least square problem for each tree.


Code Structure
--------------

**treehfd** essentially relies on `xgboost`, `numpy`, `scipy`, and `scikit-learn`.

The main class is **XGBTreeHFD**, which fits the TreeHFD decomposition from a pretrained
xgboost model. **XGBTreeHFD** sequentially calls the class **TreeHFD** to fit the decompostion
for each tree one by one, by solving a quadratic program, built using the modules
**tree_structure**, **cartesian_partition**, and **optimization_matrix**. 
The **tree_structure** module provides functions to extract the required information from
a tree of the pretrained xgboost model, that is the splitting variables and values,
the children nodes, and the variable interactions.
The **cartesian_partition** module computes the Cartesian tree partitions from the 
tree structures. 
The **optimization_matrix** module builds the matrix and vector of the quadratic
program to solve to get the decomposition coefficients for each tree.
Finally, the **validation** module checks the user inputs.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   ensemble.rst
   tree.rst
   tree_structure.rst
   cartesian_partition.rst
   optimization_matrix.rst
   validation.rst

