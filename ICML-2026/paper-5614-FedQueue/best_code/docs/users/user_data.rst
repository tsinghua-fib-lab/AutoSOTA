How to load datasets
====================

Client needs to load their local private dataset by providing a function that returns the training and validation datasets as ``torch.utils.data.Dataset`` objects. In ``APPFL``, we created a simple data class available at ``appfl.misc.data.Dataset`` that takes ``data_input`` and ``data_label`` as ``torch.Tensor`` objects. We expect in most cases this simple class would be sufficient. However, users can create more sophisticated dataset class for their own customization.

For example, suppose that we define a following function to load the dataset:

.. code-block:: python

    def get_my_local_dataset(
        **kwargs
    ):
        ...
        return train_dataset, val_dataset

Then we can load the dataset by providing the absolute/relative path to the function definition file, function name, and the keyword arguments in the client configuration file as follows:

.. code-block:: yaml

    # Local dataset
    data_configs:
    dataset_path: <path_to_you_dataset_fn>.py
    dataset_name: "get_my_local_dataset"
    dataset_kwargs:
        ...
