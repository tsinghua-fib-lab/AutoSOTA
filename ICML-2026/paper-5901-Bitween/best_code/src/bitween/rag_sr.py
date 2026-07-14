"""
Paper: RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression
OpenReview: https://openreview.net/forum?id=NdHka08uWn
Code retrieved from https://github.com/hengzhe-zhang/RAG-SR @ 8fb7eb0
"""

import numpy as np
from evolutionary_forest.forest import EvolutionaryForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sympy import parse_expr, preorder_traversal


class RAGSRRegressor(BaseEstimator, RegressorMixin):
    """RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression

    A wrapper for the EvolutionaryForestRegressor that implements the RAG-SR approach.

    Parameters
    ----------
    n_gen : int, default=100
        Number of generations for the evolutionary algorithm.
    n_pop : int, default=200
        Population size for the evolutionary algorithm.
    gene_num : int, default=10
        Number of trees (features) in each solution.
    neural_pool : float, default=0.1
        Probability of using neural generation vs retrieval.
    neural_pool_num_of_functions : int, default=5
        Maximum number of functions in generated trees.
    weight_of_contrastive_learning : float, default=0.05
        Weight of contrastive loss in the neural network.
    neural_pool_dropout : float, default=0.1
        Dropout rate for the neural network.
    neural_pool_transformer_layer : int, default=1
        Number of transformer layers in the neural network.
    neural_pool_hidden_size : int, default=64
        Hidden size of the neural network.
    neural_pool_mlp_layers : int, default=3
        Number of MLP layers in the neural network.
    selective_retrain : bool, default=True
        Whether to selectively retrain the neural network.
    negative_data_augmentation : bool, default=True
        Whether to use scale-invariant data augmentation.
    select : str, default="AutomaticLexicase"
        Selection method for the evolutionary algorithm.
    """

    def __init__(
        self,
        n_gen=100,
        n_pop=200,
        gene_num=10,
        neural_pool=0.1,
        neural_pool_num_of_functions=5,
        weight_of_contrastive_learning=0.05,
        neural_pool_dropout=0.1,
        neural_pool_transformer_layer=1,
        neural_pool_hidden_size=64,
        neural_pool_mlp_layers=3,
        selective_retrain=True,
        negative_data_augmentation=True,
        select="AutomaticLexicase",
        cross_pb=0.9,
        mutation_pb=0.1,
        max_height=10,
        basic_primitives="Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,RSin,RCos,Max,Min,Neg",
        base_learner="RidgeCV",
        normalize="MinMax",
        categorical_encoding="Target",
        **kwargs,
    ):
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.gene_num = gene_num
        self.neural_pool = neural_pool
        self.neural_pool_num_of_functions = neural_pool_num_of_functions
        self.weight_of_contrastive_learning = weight_of_contrastive_learning
        self.neural_pool_dropout = neural_pool_dropout
        self.neural_pool_transformer_layer = neural_pool_transformer_layer
        self.neural_pool_hidden_size = neural_pool_hidden_size
        self.neural_pool_mlp_layers = neural_pool_mlp_layers
        self.selective_retrain = selective_retrain
        self.negative_data_augmentation = negative_data_augmentation
        self.select = select
        self.cross_pb = cross_pb
        self.mutation_pb = mutation_pb
        self.max_height = max_height
        self.basic_primitives = basic_primitives
        self.base_learner = base_learner
        self.normalize = normalize
        self.categorical_encoding = categorical_encoding
        self.kwargs = kwargs

        # Initialize the underlying regressor
        self._initialize_regressor()

    def _initialize_regressor(self):
        """Initialize the underlying EvolutionaryForestRegressor with RAG-SR parameters."""

        # Neural network parameters
        nn_parameters = {
            "neural_pool": self.neural_pool,
            "neural_pool_num_of_functions": self.neural_pool_num_of_functions,
            "weight_of_contrastive_learning": self.weight_of_contrastive_learning,
            "neural_pool_dropout": self.neural_pool_dropout,
            "neural_pool_transformer_layer": self.neural_pool_transformer_layer,
            "neural_pool_hidden_size": self.neural_pool_hidden_size,
            "neural_pool_mlp_layers": self.neural_pool_mlp_layers,
            "selective_retrain": self.selective_retrain,
            "negative_data_augmentation": self.negative_data_augmentation,
            "negative_local_search": False,
        }

        # Initialize the regressor
        self.regressor_ = EvolutionaryForestRegressor(
            n_gen=self.n_gen,
            n_pop=self.n_pop,
            select=self.select,
            cross_pb=self.cross_pb,
            mutation_pb=self.mutation_pb,
            max_height=self.max_height,
            ensemble_size=1,  # Use a single model
            initial_tree_size="0-6",
            gene_num=self.gene_num,
            basic_primitives=self.basic_primitives,
            base_learner=self.base_learner,
            ridge_alphas="Auto",
            verbose=False,
            boost_size=None,  # No boosting
            normalize=self.normalize,
            external_archive=1,
            max_trees=10000,
            library_clustering_mode="Worst",
            pool_addition_mode="Smallest~Auto",
            pool_hard_instance_interval=10,
            random_order_replacement=True,
            pool_based_addition=True,
            semantics_length=50,
            change_semantic_after_deletion=True,
            include_subtree_to_lib=True,
            library_updating_mode="Recent",
            categorical_encoding=self.categorical_encoding,
            root_crossover=True,
            scaling_before_replacement=False,
            score_func="R2",
            number_of_invokes=0,
            mutation_scheme="uniform-plus",
            environmental_selection=None,
            record_training_data=False,
            complementary_replacement=False,
            validation_size=0,
            constant_type="Float",
            full_scaling_after_replacement=False,
            **nn_parameters,
            **self.kwargs,
        )

    def fit(self, X, y, **kwargs):
        """Fit the RAG-SR model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self.regressor_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Predict using the RAG-SR model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        return self.regressor_.predict(X)

    def model(self):
        """Get the symbolic expression of the fitted model.

        Returns
        -------
        model_str : str
            String representation of the symbolic model.
        """
        return self.regressor_.model()

    def complexity(self):
        """Calculate the complexity of the model.

        Returns
        -------
        complexity : int
            Number of nodes in the symbolic expression.
        """
        return len(list(preorder_traversal(parse_expr(self.model()))))

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) with respect to y.
        """
        return self.regressor_.score(X, y)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Initialize and fit model
    model = RAGSRRegressor(n_gen=10, n_pop=50)  # Small values for demonstration
    model.fit(X_train, y_train, categorical_features=np.zeros(X.shape[1]))

    # Evaluate
    score = model.score(X_test, y_test)
    complexity = model.complexity()

    print(f"Model: {model.model()}")
    print(f"Complexity: {complexity}")
    print(f"R² Score: {score:.4f}")
