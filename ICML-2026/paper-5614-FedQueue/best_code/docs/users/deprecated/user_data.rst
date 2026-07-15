How to define datasets
======================

User-defined dataset can be created simply using the class defined below by providing `data_input` and `data_label` as `torch` tensors.
We expect in most cases this simple class would be sufficient. However, users can create more sophisticated dataset class for their own customization.

.. autoclass:: appfl.misc.data.Dataset
    :exclude-members: __len__, __getitem__
