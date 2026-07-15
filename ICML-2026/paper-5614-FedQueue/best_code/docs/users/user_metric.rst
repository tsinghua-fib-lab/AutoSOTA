How to define metric
====================

Users can define custom evaluation metric functions for their applications. The metric function should take the model output and label as input and return the metric value. The metric value should be a scalar. For example, we can define the accuracy metric as follows:


.. code-block:: python

    import numpy as np

    def accuracy(y_true, y_pred):
        '''
        y_true and y_pred are both of type np.ndarray
        y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
        y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
        '''
        if len(y_pred.shape) == 1:
            y_pred = np.round(y_pred)
        else:
            y_pred = y_pred.argmax(axis=1)
        return 100*np.sum(y_pred==y_true)/y_pred.shape[0]

To use the defined metric function during the FL experiment, you need to provide the absolute/relative path to the metric definition file and the name of the metric function. For example, to use the accuracy metric defined above, you can add the following lines to the server configuration file, where ``do_validation`` means to perform validation after each local training round, ``do_pre_validation`` means to perform validation before each local training round (i.e., validate the global model).

.. code-block:: yaml

    client_configs:
        train_configs:
            ...
            # Client validation
            do_validation: True
            do_pre_validation: True
            metric_path: "<path_to_acc_metric>.py"
            metric_name: "accuracy"
        ...
