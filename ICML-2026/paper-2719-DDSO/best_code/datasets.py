from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
import torch
import random
from pdb import set_trace as bb
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    
class ClassificationDataModule(LightningDataModule):
    """Datamodule for classification datasets.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: str = "datasets/",
        random_seed: Optional[int] = 1,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        batch_size: int = 64,
        test_batch_size: int = 128,
        normalize: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        assert np.isclose(sum(train_val_test_split), 1), f"Train_val_test_split must sum to 1. Got {train_val_test_split} with sum {sum(train_val_test_split):0.5f}."

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.input_size: int = classification_shapes[dataset_name][0]
        self.label_size: int = classification_shapes[dataset_name][1]
        # self._generator = torch.Generator().manual_seed(random_seed)

        # self.setup()

    @property
    def dataset_type(self) -> str:
        return "regression"

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            seed_everything(self.hparams.random_seed)
            loader_fun = classification_load_funs[self.hparams.dataset_name]
            X, y = loader_fun(self.hparams.data_dir)
            if self.hparams.normalize:
                std = X.std(axis=0)
                zeros = np.isclose(std, 0.)
                X[:, ~zeros] = (X[:, ~zeros] - X[:, ~zeros].mean(axis=0)) / X[:, ~zeros].std(axis=0)
                X[:, zeros] = 0.
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            full_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
            
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=self.hparams.train_val_test_split[2], random_state=self.hparams.random_seed)
            indices = np.arange(len(X))
            train_val_idx, test_idx = next(sss1.split(X, y))

            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=self.hparams.train_val_test_split[1] / (self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1]), random_state=self.hparams.random_seed)
            train_idx, val_idx = next(sss2.split(X[train_val_idx], y[train_val_idx]))

            train_indices = train_val_idx[train_idx]
            val_indices = train_val_idx[val_idx]

            self.data_train = Subset(full_dataset, train_indices)
            self.data_val = Subset(full_dataset, val_indices)
            self.data_test = Subset(full_dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)



def _load_wdbc(data_dir):
    """
    Attribute Information:
    1) ID number
    2) Diagnosis (M = malignant, B = benign)
    3-32) Ten real-valued features are computed for each cell nucleus:

      a) radius (mean of distances from center to points on the perimeter)
      b) texture (standard deviation of gray-scale values)
      c) perimeter
      d) area
      e) smoothness (local variation in radius lengths)
      f) compactness (perimeter^2 / area - 1.0)
      g) concavity (severity of concave portions of the contour)
      h) concave points (number of concave portions of the contour)
      i) symmetry 
      j) fractal dimension ("coastline approximation" - 1)
    """
    data_file = os.path.join(data_dir, 'classification/wdbc/wdbc.data')
    colnames = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
    data = pd.read_csv(data_file, names=colnames, header=None)
    data.drop('id',axis=1,inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

    X = data.to_numpy()[:, 1:]
    y = data.to_numpy()[:, 0]
    return X, y

def _load_adult(data_dir):
    """
    Attribute Information:
    The dataset contains 16 columns
    Target filed: Income
    -- The income is divide into two classes: <=50K and >50K
    Number of attributes: 14
    -- These are the demographics and other features to describe a person
    """
    data_file = os.path.join(data_dir, 'classification/adult/adult.data')
    colnames = ["age","workclass","fnlwgt","education","educational-num","marital-status","occupation","relationship","race","gender","capital-gain","capital-loss","hours-per-week","native-country","income"]
    data = pd.read_csv(data_file, header=None, names=colnames, skipinitialspace=True)
    data = data.replace("?", np.NaN).dropna()
    category_col =['workclass', 'education','marital-status', 'occupation',
                  'relationship', 'race', 'gender', 'native-country']
    b, c = np.unique(data['income'], return_inverse=True) 
    data['income'] = c

    def encode_and_bind(original_dataframe, feature_to_encode):
      dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
      res = pd.concat([original_dataframe, dummies], axis=1)
      res = res.drop([feature_to_encode], axis=1)
      return res
 
    for feature in category_col:
        data = encode_and_bind(data, feature)

    y = data['income'].to_numpy()
    data = data.drop('income', axis=1)
    X = data.to_numpy().astype(float)
    return X, y

def _load_heart_disease(data_dir):
    """
    Column Descriptions:
        id (Unique id for each patient)
        age (Age of the patient in years)
        origin (place of study)
        sex (Male/Female)
        cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
        trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
        chol (serum cholesterol in mg/dl)
        fbs (if fasting blood sugar > 120 mg/dl)
        restecg (resting electrocardiographic results)
        -- Values: [normal, stt abnormality, lv hypertrophy]
        thalach: maximum heart rate achieved
        exang: exercise-induced angina (True/ False)
        oldpeak: ST depression induced by exercise relative to rest
        slope: the slope of the peak exercise ST segment
        ca: number of major vessels (0-3) colored by fluoroscopy
        thal: [normal; fixed defect; reversible defect]
        num: the predicted attribute
    """
    data_file = os.path.join(data_dir, 'classification/heart_disease/heart_disease_uci.csv')
    data = pd.read_csv(data_file)
    data = data.drop(['id', 'dataset'], axis=1)
    data.dropna(inplace = True)

    data.fbs = data.fbs.astype(int)
    data.exang = data.exang.astype(int)

    category_col = ['sex', 'cp', 'restecg', 'slope', 'thal']

    def encode_and_bind(original_dataframe, feature_to_encode):
      dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
      res = pd.concat([original_dataframe, dummies], axis=1)
      res = res.drop([feature_to_encode], axis=1)
      return res

    for feature in category_col:
        data = encode_and_bind(data, feature)

    y = data['num'].to_numpy()
    data = data.drop('num', axis=1)
    X = data.to_numpy().astype(float)
    return X, y

def _load_online_shoppers(data_dir):
    """
    Data Set Information:

    The dataset consists of feature vectors belonging to 12,330 sessions.
    The dataset was formed so that each session
    would belong to a different user in a 1-year period to avoid
    any tendency to a specific campaign, special day, user
    profile, or period.

    Dataset Origin:
    https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
    """
    data_file = os.path.join(data_dir, 'classification/online_shoppers/online_shoppers_intention.csv')
    data = pd.read_csv(data_file)
    data.dropna(inplace = True)

    data.Revenue = data.Revenue.astype(int)
    data.Weekend = data.Weekend.astype(int)

    category_col = ['Month', 'VisitorType']

    def encode_and_bind(original_dataframe, feature_to_encode):
      dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
      res = pd.concat([original_dataframe, dummies], axis=1)
      res = res.drop([feature_to_encode], axis=1)
      return res

    for feature in category_col:
        data = encode_and_bind(data, feature)

    y = data['Revenue'].to_numpy()
    data = data.drop('Revenue', axis=1)
    X = data.to_numpy().astype(float)
    return X, y

def _load_dry_bean(data_dir):
    """
    Attribute Information:
    1.) Area (A): The area of a bean zone and the number of pixels within its boundaries.
    2.) Perimeter (P): Bean circumference is defined as the length of its border.
    3.) Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
    4.) Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
    5.) Aspect ratio (K): Defines the relationship between L and l.
    6.) Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
    7.) Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
    8.) Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
    9.) Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
    10.) Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
    11.) Roundness (R): Calculated with the following formula: (4piA)/(P^2)
    12.) Compactness (CO): Measures the roundness of an object: Ed/L
    13.) ShapeFactor1 (SF1)
    14.) ShapeFactor2 (SF2)
    15.) ShapeFactor3 (SF3)
    16.) ShapeFactor4 (SF4)
    17.) Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)
    """
    data_file = os.path.join(data_dir, 'classification/dry_bean/dry_bean_dataset.csv')
    data = pd.read_csv(data_file)
    data.dropna(inplace = True)

    b, c = np.unique(data['Class'], return_inverse=True) 
    data['Class'] = c

    y = data['Class'].to_numpy()
    data = data.drop('Class', axis=1)
    X = data.to_numpy().astype(float)
    return X, y


classification_load_funs = {
    "wdbc": _load_wdbc,
    "adult": _load_adult,
    "heart-disease": _load_heart_disease,
    "online-shoppers": _load_online_shoppers,
    "dry-bean": _load_dry_bean
}

classification_shapes = {
    "wdbc": (30, 2),
    "adult": (104, 2),
    "heart-disease": (23, 5),
    "online-shoppers": (28, 2),
    "dry-bean": (16, 7)
}

