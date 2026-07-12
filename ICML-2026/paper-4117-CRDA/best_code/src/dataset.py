from torch.utils.data import Dataset
from .baseline import BaselineRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional

class AbstractDataset(Dataset):
    def __init__(self, name, file_path: str, sample_size: Optional[list[int]] = None, seed: Optional[int] = None):
        """
        Initialize the AbstractDataset.
        
        Args:
            name: Name identifier for the dataset
            file_path (str): Path to the CSV file containing the dataset
            sample_size (Optional[list[int]]): Number of samples to randomly select from the dataset.
                                             If None, uses the entire dataset.
            seed (Optional[int]): Random seed for reproducible sampling and train/test splits
            
        Raises:
            ValueError: If sample_size is greater than the number of rows in the dataset
        """
        self.name = name
        self.file_path = file_path
        self.seed = seed
        self.df = pd.read_csv(file_path)

        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        if sample_size is not None:
            if sample_size > len(self.df): raise ValueError(f"Sample size {sample_size} is greater than the number of rows in the dataset {self.name}: {len(self.df)}.")
            self.df = self.df.sample(sample_size, random_state=seed)
            self.df = self.df.reset_index(drop=True)

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1:] # last column

        self.X_preprocessed = None
        self.y_preprocessed = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.all_residuals = None
        self.train_residuals = None
        self.test_residuals = None

    def preprocess(self):
        """
        Preprocess the dataset by standardizing features and normalizing the target variable.
        
        Features (X) are standardized using StandardScaler (mean=0, std=1).
        Target variable (y) is normalized using MinMaxScaler (range [0,1]).
        
        Returns:
            tuple: A tuple containing:
                - X_preprocessed (numpy.ndarray): Standardized feature matrix
                - y_preprocessed (numpy.ndarray): Normalized target variable array
        """
        # Standardize the features
        self.scaler_X = StandardScaler()
        self.X_preprocessed = self.scaler_X.fit_transform(self.X)

        # Normalize the target variable
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y).ravel()

        return self.X_preprocessed, self.y_preprocessed
    
    def split(self, test_size=0.2, X=None, y=None, seed=None):
        """
        Split the dataset into training and testing sets.
        
        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split.
                                       Defaults to 0.2 (20%).
            X (numpy.ndarray, optional): Feature matrix to split. If None, uses preprocessed features.
            y (numpy.ndarray, optional): Target variable to split. If None, uses preprocessed target.
            seed (int, optional): Random seed for the split. If None, uses the dataset's seed.
            
        Returns:
            tuple: A tuple containing:
                - X_train (numpy.ndarray): Training feature matrix
                - X_test (numpy.ndarray): Testing feature matrix  
                - y_train (numpy.ndarray): Training target variable array
                - y_test (numpy.ndarray): Testing target variable array
                
        Raises:
            AssertionError: If data hasn't been preprocessed and no X, y provided
        """
        if X is None and y is None:
            assert self.y_preprocessed is not None and self.X_preprocessed is not None, "Preprocess the data first."
            X = self.X_preprocessed
            y = self.y_preprocessed

        if seed is None:
            seed = self.seed

        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)
        self.y_train = y_train.ravel()
        self.y_test = y_test.ravel()
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def add_residuals(self, model: BaselineRegressor):
        """
        Calculate and store residuals from a baseline regressor model.
        
        Residuals are computed as actual values minus predicted values.
        Calculates residuals for the full dataset, training set, and test set.
        
        Args:
            model (BaselineRegressor): A fitted baseline regression model that implements
                                     a predict() method
                                     
        Returns:
            tuple: A tuple containing:
                - all_residuals (numpy.ndarray): Residuals for the entire preprocessed dataset
                - train_residuals (numpy.ndarray): Residuals for the training set
                - test_residuals (numpy.ndarray): Residuals for the test set
        """
        self.all_residuals = self.y_preprocessed - model.predict(self.X_preprocessed)
        
        self.train_residuals = self.y_train - model.predict(self.X_train)
        self.test_residuals = self.y_test - model.predict(self.X_test)

        return self.all_residuals, self.train_residuals, self.test_residuals
