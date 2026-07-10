This folder contains the code files used in the paper titled "Min-Max Optimisation for Nonconvex-Nonconcave Functions Using a Random Zeroth-Order Extragradient Algorithm".

Link to the paper: "https://openreview.net/pdf?id=1bxY1uAXyr".

In this project, we try to solve the minimax problem considering functions satisfying the weak minty variational inequalities using the Gaussian random zeroth-order oracles and a zeorth-order extra grdaient algorithm in constrained, unconstrained, and non-smooth settings.

This folder contains five Python notebook files:

"ZOEG_low.ipynb" and "DS_Low.ipynb" include the implementations of the ZOEG and DS algorithms, respectively, for the first three numerical examples: (1) low dimensional toy problems, (2) robust least squares problem, and (3) data poisoning attack to logistic regression.

"ZOEG_NN2.ipynb" and "DR_NN2.ipynb" provide the implementations of the ZOEG and DS algorithms for the fourth numerical example (robust optimisation).

"ZOEG_LaneMerge.ipynb" contains the code for the fifth numerical example presented in the paper, which addresses the lane merging problem.
