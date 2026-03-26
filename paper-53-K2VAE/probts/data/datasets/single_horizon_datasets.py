# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from torch.utils.data import IterableDataset
from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cyclic
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    SelectFields,
    Transformation,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler,
    ExpectedNumInstanceSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    TransformedDataset,
)
from probts.data.data_utils.time_features import fourier_time_features_from_frequency, AddCustomizedTimeFeatures


class SingleHorizonDataset():
    """
    SingleHorizonDataset: Handles dataset transformation and instance splitting for single-horizon forecasting tasks.

    Parameters:
    ----------
    input_names : list
        List of input field names required by the model.
    history_length : int
        Length of the historical time series window for input data.
    prediction_length : int
        Length of the forecasting horizon.
    freq : str
        Data frequency (e.g., 'H' for hourly, 'D' for daily).
    multivariate : bool, optional, default=True
        Indicates if the dataset contains multiple target variables.
    """
    def __init__(
        self,
        input_names: list,
        history_length: int,
        context_length: int,
        prediction_length: int,
        freq: str,
        multivariate: bool = True
    ):
        super().__init__()
        self.input_names_ = input_names
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        if multivariate:
            self.expected_ndim = 2
        else:
            self.expected_ndim = 1

    def get_sampler(self):
        """
        Creates samplers for training, validation, and testing.
        - Training: Generates instances randomly.
        - Validation and Testing: Always selects the last time point.
        """
        # returns a set of indices at which training instances will be generated
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=self.history_length,
            min_future=self.prediction_length,
        )

        self.val_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.prediction_length,
        )
        
        self.test_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.prediction_length,
        )


    def create_transformation(self, data_stamp=None) -> Transformation:
        """
        Creates a data transformation pipeline to prepare inputs for the model.
        Adds features such as time attributes and observed value indicators.

        Parameters:
        ----------
        data_stamp : np.array, optional
            Precomputed time features. If None, features are generated based on the data frequency.

        Returns:
        ----------
        Chain : Transformation
            A chain of transformations applied to the dataset.
        """
        if data_stamp is None:
            if self.freq in ["M", "W", "D", "B", "H", "min", "T"]:
                time_features = fourier_time_features_from_frequency(self.freq)
            else:
                time_features = fourier_time_features_from_frequency('D')
            self.time_feat_dim = len(time_features) * 2
            time_feature_func = AddTimeFeatures
        else:
            self.time_feat_dim = data_stamp.shape[-1]
            time_features = data_stamp
            time_feature_func = AddCustomizedTimeFeatures

        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=self.expected_ndim,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                time_feature_func(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str, auto_search=False):
        """
        Creates an instance splitter for training, validation, or testing.

        Parameters:
        ----------
        mode : str
            Mode of operation. Must be one of ['train', 'val', 'test'].

        Returns:
        ----------
        InstanceSplitter : Transformation
            A splitter transformation that slices input data for model training or evaluation.
        """
        assert mode in ["train", "val", "test"]

        self.get_sampler()
        instance_sampler = {
            "train": self.train_sampler,
            "val": self.val_sampler,
            "test": self.test_sampler,
        }[mode]

        if auto_search:
            past_length = self.context_length + self.prediction_length
        else:
            past_length=self.history_length
        
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def get_iter_dataset(self, dataset: Dataset, mode: str, data_stamp=None, auto_search=False) -> IterableDataset:
        """
        Creates an iterable dataset for training, validation, or testing.

        Parameters:
        ----------
        dataset : Dataset
            Input dataset to transform.
        mode : str
            Mode of operation. Must be one of ['train', 'val', 'test'].
        data_stamp : np.array, optional
            Precomputed time features.

        Returns:
        ----------
        IterableDataset : TransformedIterableDataset
            Transformed dataset with applied transformations and instance splitting.
        """
        assert mode in ["train", "val", "test"]

        transform = self.create_transformation(data_stamp)
        if mode == 'train':
            with env._let(max_idle_transforms=100):
                instance_splitter = self.create_instance_splitter(mode)
        else:
            instance_splitter = self.create_instance_splitter(mode, auto_search=auto_search)


        input_names = self.input_names_

        iter_dataset = TransformedIterableDataset(
            dataset,
            transform=transform
            + instance_splitter
            + SelectFields(input_names),
            is_train=True if mode == 'train' else False
        )

        return iter_dataset



class TransformedIterableDataset(IterableDataset):
    """
    A transformed iterable dataset that applies a transformation pipeline on-the-fly.

    Parameters:
    ----------
    dataset : Dataset
        The original dataset to transform.
    transform : Transformation
        The transformation pipeline to apply.
    is_train : bool, optional, default=True
        Whether the dataset is used for training.
    """
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True
    ):
        super().__init__()

        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if is_train else dataset,
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        return iter(self.transformed_dataset)