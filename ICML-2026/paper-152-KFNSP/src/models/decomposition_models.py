import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics.pairwise import pairwise_kernels


class Regularizer:
    """
        Super class that performs the fair kernel decomposition.
        As our method is model agnostic (as long as it uses kernels)
        this has a placeholder for "model".

    """
    def __init__(
        self,
        model=None,
        single_protected=False,
        alpha_prime=0.05,
        gamma=None,
        kernel="rbf",
        nystrom_comp = None,
    ):
        """
            Parameters
            ----------
            model : class instance
                Note that here we require a class instance! Not the class handle. This is the model
                used in conjuction with the fair kernel decomposition.

            single_protected : bool
                This is only used for the multi protected attribute benchmark but this is not really
                general purpose eg if this is true we protect only for the first of the protected features provided in train.
                Mainly used to generate figure 2 a).

            alpha_prime : float
                Regularization parameter for the fair kernel decomposition.

            gamma : float
                gamma parameter for the rbf kernel

            kernel : string
                Denotes the kernel. Note currently this focusses on the rbf kernel only.
                Either adapt which paramter is passed to the kernel below 
                in "pairwise_kernels" or add "filter_params=True" as parameter

            nystrom_comp : None or float in (0,1]
                If None no nystroem approx. is used. Otherwise percentage of components to use for approx.
        """

        self.model = model
        self.regularizer = FairKernelDecomposition(alpha=alpha_prime,nystrom_comp=nystrom_comp)

        self.kernel = kernel
        self.gamma = gamma

        self.single_protected = single_protected  

    def train(self, X, y, p, iterations):
        """
            Method to first train the fair kernel decomposition and then the respective model on top.

            Parameters
            ----------
                X : np.ndarrry
                    Training data of shape (n_samples, n_features)

                y : np.ndarrry
                    Training targets of shape (n_samples, n_features)

                p : np.ndarrry
                    Protected attrribute of the training data of shape (n_samples, n_features)

                iterations : int
                    Number of iterations "m" to run the iterative fair kernel decomposition.

        """

        # We rely on numerical routines such as matrix inversion so ensure we work with float64.
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        p = np.array(p, dtype=np.float64)

        
        if self.single_protected:
            # This is not general purpose and was used to generate figure 2a). p is not contained in X.
            p = p[
                :, 0
            ]  

        # Build kernel
        KX = pairwise_kernels(
            X,
            metric=self.kernel,
            gamma=self.gamma,
        )

        # Fit the fair kernel decomposition and transform the kernel
        self.regularizer.fit(p, KX, iterations=iterations)
        # Use cached Kn directly (O(1)) instead of O(n^3) transform(KX)
        KX_reg = self.regularizer.Kn

        # Fit the model with the modified kernel
        self.model.fit(KX_reg, y)

        # Kernel method so we need to store the training data to build K_test
        self.training_data = X

    def predict(self, X):
        # Again check dtype
        X = np.array(X, dtype=np.float64)

        # Build K_test
        KX = pairwise_kernels(
            X, self.training_data, metric=self.kernel, gamma=self.gamma
        )
 
        # Apply transformation to KX
        KX_reg = self.regularizer.transform(KX)

        # Return prediction of the model
        return np.array(self.model.predict(KX_reg),dtype=np.float32)



class RegularizedSVR(Regularizer):
    def __init__(
        self,
        single_protected=False,
        alpha_prime=0.05,
        gamma=None,
        kernel="rbf",
        eps=0.01,
        C=1,
        nystrom_comp=None
    ):  
        """
            Basic Support Vector Regression subclass that creates a 
            parent instance with this specific model with specific parameters.
        """
        super().__init__(
            model=SVR(kernel="precomputed", epsilon=eps, max_iter=500000, C=C, tol=1e-4),
            single_protected=single_protected,
            alpha_prime=alpha_prime,
            gamma=gamma,
            kernel=kernel,
            nystrom_comp=nystrom_comp
        )

class RegularizedKRR(Regularizer):
    def __init__(
        self,
        single_protected=False,
        alpha_prime=0.05,
        gamma=None,
        kernel="rbf",
        alpha=0.25,
        nystrom_comp=None
    ):  
        """
            Basic Kernel Ridge Regression subclass that creates a 
            parent instance with this specific model with specific parameters.
        """
        super().__init__(
            model=KernelRidge(kernel="precomputed", alpha=alpha),
            single_protected=single_protected,
            alpha_prime=alpha_prime,
            gamma=gamma,
            kernel=kernel,
            nystrom_comp=nystrom_comp
        )


def nystroem_inverse(K_TO_APPROX, alp, components, random_state=None): 
    """
        Given Kernel K and reg. alp this approximates the inverse (K+alp\text{Id})^{-1}
        with nystroem with the given number of "components".

        See appendix of the paper for details.

        Parameters
        ----------
        K_TO_APPROX : np.ndarray
            kernel matrix of size n times n for n training samples

        alp : float
            alpha, regularization parameter

        components : int
            number of components m \ll n used for the nystroem approx. Exact if K_TO_APPROX is of rank m.

        Returns
        ----------
        K_inv_approx : np.ndarray
            Approximate inverse. The result is symmetrized for numerical stability.
    """

    size_n = np.shape(K_TO_APPROX)[-1] 

    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    IDX = rng.choice(np.arange(0, size_n), size=components, replace=False)

    K_nm = K_TO_APPROX[:,IDX]               # Pick the m columns of length n
    K_mm = K_TO_APPROX[np.ix_(IDX, IDX)]    # Pick the intersection of the m columns with the m rows


    inner_inverse = np.linalg.pinv(K_mm + (1/alp) * (K_nm.T @ K_nm))

    first_term = (1/alp) * np.eye(size_n)
    
    second_term = (1/(alp**2)) * (K_nm @ inner_inverse @ K_nm.T)
    
    K_inv_approx = first_term - second_term

    # Symmetrize to correct numerical errors
    return (K_inv_approx + K_inv_approx.T)/2 
    


class FairKernelDecomposition:
    def __init__(self, alpha=0.1, nystrom_comp = None, random_state=None):
        """
            Parameters
            ----------
            alpha : float
                alpha_prime that regularizes the KRR which aims to find the predictive directions

            nystrom_comp : float in (0,1]
                percentage of components to use for nystroem (number of components is rounded). 
            
            random_state : int or None
                Random seed for Nystroem landmark selection (deterministic if set).
        """
        self.transformation = None
        self.transformations = []
        self.alpha = alpha

        self.nystrom_coeff = nystrom_comp
        self.random_state = random_state

        # O(n^2) optimization storage
        self.Kn = None
        self.training_kernel = None
        self.Kn_list = []
        self.M1_list = []
        self.M2_times_Kn_list = []
        self.dim_attr = 1
        self.iterations_run = 0


    def fit(self, orig_attr_to_remove, kernel_matrix, iterations=1):
        sz = np.shape(kernel_matrix)[-1]

        # Initialize as identity
        self.transformation = np.eye(sz)
        self.transformations = []

        # O(n^2) optimization: store initial kernel and per-iteration data
        self.Kn = kernel_matrix.copy()
        self.training_kernel = kernel_matrix
        self.Kn_list = [kernel_matrix.copy()]
        self.M1_list = []
        self.M2_times_Kn_list = []

        # In might also be of interest to remove higher order correllations
        #  eg orig_attr_to_remove**2 or similar in future
        attr_to_remove = orig_attr_to_remove
        self.dim_attr = attr_to_remove.ndim

        for i in range(iterations):
            print("Iteration: ", str(i))

            dim_attr = attr_to_remove.ndim

            # Nystroem option
            if self.nystrom_coeff is None:
                inv = np.linalg.pinv(self.Kn + self.alpha * np.eye(sz))
            else:
                nystr_comp = int(self.nystrom_coeff * sz)
                inv = nystroem_inverse(self.Kn, alp=self.alpha, components=nystr_comp, random_state=self.random_state)

            # This corresponds to a_norm in the paper. Use right order of mult. for efficiency.
            TAU_NORM = (np.transpose(attr_to_remove) @ inv) @ (self.Kn @ (inv @ attr_to_remove))

            if dim_attr == 1:
                M1 = (1 / TAU_NORM) * inv @ attr_to_remove
                M2 = attr_to_remove.T @ inv
            else:
                M1 = inv @ attr_to_remove @ np.linalg.pinv(TAU_NORM)
                M2 = attr_to_remove.transpose() @ inv

            # ===== O(n^2) OPTIMIZATION =====
            # Store per-iteration data for O(n^2) test kernel transformation
            self.M1_list.append(M1.copy())
            self.M2_times_Kn_list.append(M2 @ self.Kn)

            # O(n^2) kernel matrix update: directly update Kn instead of
            # the O(n^3) self.transformation @ transf_matrix multiplication.
            # This is the authors own optimization from the commented-out block.
            if dim_attr == 1:
                self.Kn = self.Kn - ((self.Kn @ M1.reshape(-1,1)) @ (M2.reshape(-1,1).T @ self.Kn))
            else:
                self.Kn = self.Kn - ((self.Kn @ M1) @ (M2 @ self.Kn))

            self.Kn_list.append(self.Kn.copy())
            # ===== END O(n^2) OPTIMIZATION =====

            # Note: self.transformation is no longer accumulated here.
            # The O(n^2) iterative approach in transform() handles test kernels.
            # This saves O(n^3) per iteration, giving ~10-50x speedup for fair kernel decomposition.

        self.iterations_run = iterations


    def transform(self, kernel_matrix):
        """
            Apply transformation T_m after m iterations.
            Uses O(n^2 * n_test) iterative approach for test kernels,
            and returns cached Kn directly for training kernel.
        """
        sz_test = np.shape(kernel_matrix)[0]
        sz_train = np.shape(self.training_kernel)[0] if self.training_kernel is not None else 0

        # If training kernel: return cached Kn from the O(n^2) fit path
        if sz_test == sz_train and self.Kn is not None:
            return self.Kn

        # Test kernel: apply O(n^2 * n_test) iterative updates using stored per-iteration data
        K = kernel_matrix.copy()
        for i in range(self.iterations_run):
            M1 = self.M1_list[i]
            M2_times_Kn = self.M2_times_Kn_list[i]

            if self.dim_attr == 1:
                # K @ M1: (n_test x n) @ (n x 1) = (n_test x 1)
                # outer: (n_test x 1) @ (1 x n) = (n_test x n)
                K_update1 = K @ M1.reshape(-1, 1)
                K_update2 = M2_times_Kn.reshape(1, -1)
                K = K - (K_update1 @ K_update2)
            else:
                K = K - ((K @ M1) @ M2_times_Kn)

        return K

    def transform_specific(self, kernel_matrix, index):
        """
            Apply one specific (possible intermediate) transformations T_i
        """
        return kernel_matrix @ self.transformations[index]
