# Preprocessed Data

The users may need to preprocess their raw datasets. The preprocessed data can be stored in this directory.

## Example 1. Coronahack
The coronahack dataset in the ``examples/dataset/RawData`` directory consists of **images** and their associated **classes**.
As loading the raw datasets can be time-consuming, one can preprocess the raw datasets such that they can be expressed by **numbers**.

As an example, we have created a ``Coronahack_Preprocess.py`` file that converts the raw coronahack dataset (i.e., images and classes) into the preprocessed dataset (i.e., real numbers between 0 and 1 for the images, namely "data input", and integer numbers for the classes, namely "data label").

More specifically, by setting ``num_clients=4`` in ``Coronahack_Preprocess.py``, it produces 5 ``json`` files

```
all_test_data.json (dataset for a server)
all_train_data_client_0.json (dataset for a client 0)
all_train_data_client_1.json (dataset for a client 1)
all_train_data_client_2.json (dataset for a client 2)
all_train_data_client_3.json (dataset for a client 3)
```

Each ``json`` file has the following format:

```
   {
   "x": [ [data input 1], [data input 2], ..., [data input N] ],
   "y": [ data label 1, data label 2, ..., data label N ]
   }
```
