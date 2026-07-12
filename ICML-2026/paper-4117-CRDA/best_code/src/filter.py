from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from .utils.logger import Logger
from .dataset import AbstractDataset
from .utils.config import Config
from .causal import CausalGraphLearner

class Filter:
    def __init__(self, dataset: AbstractDataset, config: Config, logger: Logger):
        """
        Initialize the Filter for feature selection based on causal discovery and correlation analysis.
        
        Args:
            dataset (AbstractDataset): Preprocessed dataset containing features, target, and residuals
            config (Config): Configuration object containing experiment parameters and thresholds
            logger (Logger): Logger instance for recording information and warnings
        """
        self.dataset = dataset
        self.config = config
        self.logger = logger

        self.target = dataset.y_preprocessed
        self.features = np.hstack((dataset.X_preprocessed, dataset.all_residuals.reshape(-1, 1)))

        self.df = pd.DataFrame(self.features, columns=[f"F{i}" for i in range(self.features.shape[1]-1)] + ["Z"])
        self.df["T"] = self.target
        self.feature_names = [col for col in self.df.columns if col not in ['T', 'Z']]

    def _create_causal_graph(self):
        """
        Create a causal graph using the PC algorithm to discover causal relationships.
        
        Uses CausalGraphLearner with the configured alpha value to learn causal structure
        from the dataset. Optionally saves the graph visualization if save_plots is enabled.
        
        Returns:
            CausalGraphLearner: Fitted causal graph learner object
            
        Raises:
            Exception: If causal graph creation fails (caught by calling method)
        """
        self.logger.info(f"Creating causal graph for {self.dataset.name}")
        learner = CausalGraphLearner(alpha=self.config.alpha)
        learner.fit(self.df)

        if self.config.save_plots:
            causal_dir = os.path.join(self.config.experiment_dir, "causal_graphs")
            os.makedirs(causal_dir, exist_ok=True)     
            graph_viz = learner.plot_graph()
            graph_viz.render(os.path.join(causal_dir, f"{self.dataset.name}_causal_graph"), format="png", cleanup=True)
            self.logger.info(f"Causal graph image saved to {causal_dir}")

        return learner
    
    def _create_correlation_matrix(self):
        """
        Create and save a correlation matrix heatmap for all variables in the dataset.
        
        Generates an absolute correlation matrix heatmap showing relationships between
        all features, residuals (Z), and target (T). Saves the visualization to the
        experiment directory.
        
        Returns:
            pandas.DataFrame: Absolute correlation matrix of all variables
        """
        corr_matrix = self.df.corr().abs()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title(f"Correlation matrix for {self.dataset.name}")
        corr_dir = os.path.join(self.config.experiment_dir, "correlation_matrices")
        os.makedirs(corr_dir, exist_ok=True)     
        plt.savefig(os.path.join(corr_dir, f"{self.dataset.name}_correlation_matrix.png"))
        self.logger.info(f"Correlation matrix saved to {corr_dir}")
        return corr_matrix

    def run_checks(self):
        """
        Run comprehensive feature filtering based on causal discovery and correlation analysis.
        
        This method performs a multi-step filtering process:
        1. Creates a causal graph to identify features conditionally independent of residuals (Z)
        2. Generates correlation matrix visualization (if save_plots enabled)
        3. Checks correlation between residuals (Z) and target (T)
        4. Filters features that are not significantly correlated with residuals (Z)
        5. Finds intersection of uncorrelated features and causally non-connected features
        6. Scores candidate features based on relevance to target vs connection to residuals
        
        The scoring formula maximizes correlation with target (T) while minimizing
        correlation with residuals (Z): score = |corr(feature, T)| - |corr(feature, Z)|
        
        Returns:
            list[int] or None: List of feature indices ranked by their scores (best first).
                              Returns None if no suitable features are found or if causal
                              graph creation fails.
        """
        # === Step 1: Create causal graph and check if there are nodes that are at least conditionally independent of Z === #
        try:
            learner = self._create_causal_graph()
        except Exception as e:
            self.logger.warning(f"Error creating causal graph: {e}")
            self.logger.warning(f"Dataset might not be suitable for causal discovery")
            return None

        not_connected_to_z = learner.get_non_adjacent_nodes('Z')
        if not not_connected_to_z:
            self.logger.warning(f"No features are conditionally independent of Z")
            return None
        self.logger.info(f"Features that are not connected to Z (conditional independence): {not_connected_to_z}")
        
        # Check correlation of features with Z
        if self.config.save_plots:
            self._create_correlation_matrix()

        
        # === Step 2: Filter features not correlated with Z === #
        uncorrelated_with_z = []
        for feature in self.feature_names:
            r, p = pearsonr(self.df['Z'], self.df[feature])
            self.logger.info(f"[Z vs {feature}]  r: {r:.4f}, p: {p:.4f}")
            if abs(r) < self.config.r_corr_threshold and p > self.config.p_corr_threshold:
                uncorrelated_with_z.append(feature)

        if len(uncorrelated_with_z) == 0:
            self.logger.warning("No features are uncorrelated with Z.")
            return None

        # === Step 3: Intersect with PC-discovered non-connected nodes === #
        candidates = list(set(uncorrelated_with_z).intersection(set(not_connected_to_z)))
        self.logger.info(f"Candidates (uncorrelated + not connected to Z): {candidates}")

        # === Step 4: Score candidates (high corr with T, low corr with Z) === #
        scores = []
        for feature in candidates:
            r_t, _ = pearsonr(self.df[feature], self.df['T'])  # correlation with T
            r_z, _ = pearsonr(self.df[feature], self.df['Z'])  # correlation with Z

            score = abs(r_t) - abs(r_z)  # maximize relevance to T, minimize connection to Z
            scores.append({
                'feature': feature,
                'score': score,
                'r_with_T': r_t,
                'r_with_Z': r_z
            })

        # Convert to DataFrame for sorting and readability
        score_df = pd.DataFrame(scores).sort_values(by='score', ascending=False)
        self.logger.info("Ranked candidate features:\n" + score_df.to_string())

        best_features_in_order = [int(f[1:]) for f in score_df['feature'].tolist()]
        return best_features_in_order


