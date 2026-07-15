Data Readiness Assurance Framework in APPFL
====================================================

This guide provides a unified framework to enable **data readiness metrics**, **custom rules**, and **automated remedies** within the APPFL environment using **standard configurations** and **CADRE (Customizable Assurance of Data Readiness) modules**.

Users can choose either approach or combine both in a single experiment.

.. _overview:

Overview
--------

APPFL supports two key mechanisms for evaluating data readiness:

1. **Built-in Data Readiness Metrics and Plots** – Easily enabled via YAML configuration.

2. **CADRE Modules** – Customizable modules that define advanced metrics, rules, and remedies for specific datasets and tasks.

.. _built-in-data-readiness-metrics:

1. Built-in Data Readiness Metrics
----------------------------------

You can generate general metrics and visualizations like sample size, class imbalance, and class distributions by setting the following in the **server configuration YAML**:

.. code-block:: yaml


    client_configs:
        data_readiness_configs:
            generate_dr_report: true              # Enable/disable DR report generation
            output_dirname: "./output"            # Directory to save the report
            output_filename: "data_readiness_report"  # Output file name
            dr_metrics:                           # Metrics to evaluate
                class_imbalance: true
                sample_size: true
                ...
            plot:                                 # Visualizations
                class_distribution_plot: true
                ...


**Output**:

* ``data_readiness_report.html`` in the ``output/`` folder.
* Automatically generated **before training** during the experiment run.

.. _define-custom-cadre-modules:

2. Defining Custom Data Readiness Functions with CADRE Modules
--------------------------------------------------------------

To implement **custom data readiness metrics, rules and remedies**, users can define **CADRE modules** in Python. These modules offer full flexibility for identifying issues in the data and can even apply remedies defined in the CADRE module when enabled.

.. _writing-your-own-cadre-module:

Writing Your Own CADRE Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CADRE module is a Python class that **inherits from `BaseCADREModule`** in ``appfl.misc.data_readiness.base_cadremodule``. You define your **custom metrics**, **rules**, and **remedies** inside it.

Each module implements three core functions:

* ``metric(self, **kwargs)`` – Computes user-defined metrics based on the dataset.
* ``rule(self, metric_result, **kwargs)`` – Determines if the computed metric indicates a data readiness issue.
* ``remedy(self, metric_result, **kwargs)`` – Applies a fix to resolve the issue.

If the user is only interested in defining metrics, the ``rule`` and ``remedy`` functions can be omitted.

**Basic Template**:

.. code-block:: python


    from appfl.misc.data_readiness import BaseCADREModule

    class MyCustomCADREModule(BaseCADREModule):
        def __init__(self, train_dataset, **kwargs):
            super().__init__(train_dataset, **kwargs)

        def metric(self, **kwargs):
            # Compute and return your metric as a dictionary
            # Example: return {"my_metric1": 0.5, "my_metric2": 0.8, ...}
            pass

        def rule(self, metric_result, threshold=0.0):
            # Define the logic to check if a problem exists (optional)
            # Example: return metric_result["my_metric"] > threshold
            pass

        def remedy(self, metric_result, logger, **kwargs):
            # Apply remedy and return updated dataset in dictionary format (optional)
            # Example:
            # return {"ai_ready_dataset": self.train_dataset, "metadata": None}
            pass


**Real Example**:
A sample CADRE module for **handling class imbalance** in the MNIST dataset is available here:

``examples/resources/configs/mnist_dr/cadre_module/handle_ci.py``

It demonstrates both how to detect class imbalance and how to automatically balance the classes if remedies are enabled.

.. _configuring-in-yaml:

Configuring in YAML
~~~~~~~~~~~~~~~~~~~

Once your module is created, register it in the server config like this:

.. code-block:: yaml


    cadremodule_configs:
        cadremodule_path: ./resources/configs/mnist_dr/cadre_module/handle_ci.py
        cadremodule_name: CADREModuleCI       # Name of the class inside the .py file
        remedy_action: true                   # Apply remedy if supported


This will activate your module before training begins.

**Sample Modules**

The following issues can be detected and remedied using CADRE modules for the MNIST dataset and they are available in the ``examples/resources/configs/mnist_dr/cadre_module/`` directory. In APPFL repository it's available `here <https://github.com/APPFL/APPFL/tree/main/examples/resources/configs/mnist_dr>`_:

* Class imbalance
* Duplicate samples
* Noisy data
* Outliers
* Memory usage

.. _running-experiments:

3. Running Experiments
----------------------

**Using MPI**:

.. code-block:: bash


    mpiexec -n 3 python ./mpi/run_mpi.py \
        --server_config ./resources/configs/mnist_dr/server_fedavg_cadremodule.yaml \
        --client_config ./resources/configs/mnist_dr/client_1_cadremodule.yaml


**Using gRPC**:

.. code-block:: bash


    # Terminal 1 (Server)
    python ./grpc/run_server.py --config ./resources/configs/mnist_dr/server_fedavg_cadremodule.yaml

    # Terminal 2 (Client 1)
    python ./grpc/run_client.py --config ./resources/configs/mnist_dr/client_1_cadremodule.yaml

    # Terminal 3 (Client 2)
    python ./grpc/run_client.py --config ./resources/configs/mnist_dr/client_2_cadremodule.yaml


.. _outputs:

4. Outputs
----------

After execution, the following files will appear in the `output/` directory:

* ``data_readiness_report.html`` – Data readiness report with general metrics, plots, and CADRE module results.

.. _standalone-usage-of-data-readiness-in-appfl:

5. Standalone Usage of Data Readiness in APPFL
------------------------------------------------

APPFL provides both built-in data readiness metrics and the ability to define custom CADRE modules for issue detection and remediation. While these tools are integrated into APPFL workflows, they can also be used standalone for dataset readiness inspection and cleanup.

.. _built-in-data-readiness-metrics-example:

Built-in Data Readiness Metrics Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``appfl.misc.data_readiness.metrics`` module includes utility functions like `imbalance_degree` that can be used directly to evaluate the class imbalance of a dataset used for a classification task.

.. code-block:: python

    import torch
    from torch.utils.data import Dataset
    from appfl.misc.data_readiness.metrics import imbalance_degree

    class ToyDataset(Dataset):
        def __init__(self):
            self.data = [
                (torch.tensor([1.0, 2.0]), 0),
                (torch.tensor([1.0, 2.0]), 0),
                (torch.tensor([3.0, 4.0]), 1),
            ]

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = ToyDataset()
    labels = [label for _, label in dataset]

    # Compute class imbalance degree
    imbalance = imbalance_degree(labels)
    print("Imbalance degree:", imbalance)

.. _custom-cadre-module-example:

Custom CADRE Module Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can define their own CADRE modules by subclassing ``BaseCADREModule``. The following is a simple duplicate checker that removes repeated input samples.

.. code-block:: python

    import torch
    from torch.utils.data import Dataset
    from appfl.misc.data_readiness import BaseCADREModule

    class SimpleDuplicateChecker(BaseCADREModule):
        """
        A simple duplicate checker that removes repeated input samples.
        """
        def __init__(self, train_dataset, **kwargs):
            super().__init__(train_dataset, **kwargs)

        def metric(self, **kwargs):
            """
            Compute the proportion of duplicate samples in a dataset.
            """
            data_input = torch.stack([x for x, _ in self.train_dataset])
            counts = {}
            for sample in data_input:
                key = str(sample.tolist())
                counts[key] = counts.get(key, 0) + 1

            num_duplicates = len(data_input) - len(counts)
            return {"duplicates": round(num_duplicates / len(data_input), 2)}

        def rule(self, metric_result, threshold=0.1, **kwargs):
            """
            Check if the metric result exceeds the threshold.
            """
            return metric_result["duplicates"] > threshold

        def remedy(self, metric_result, **kwargs):
            """
            Remove duplicate samples from the dataset if the rule condition is met.
            """
            if not self.rule(metric_result):
                return {"ai_ready_dataset": self.train_dataset, "metadata": None}

            seen = set()
            cleaned = []
            for x in self.train_dataset:
                key = str(x[0].tolist())
                if key not in seen:
                    seen.add(key)
                    cleaned.append(x)

            return {"ai_ready_dataset": cleaned, "metadata": None}

    class ToyDataset(Dataset):
        def __init__(self):
            self.data = [
                (torch.tensor([1.0, 2.0]), 0),
                (torch.tensor([1.0, 2.0]), 0), # Duplicate
                (torch.tensor([3.0, 4.0]), 1),
            ]

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    # Apply the CADRE module
    dataset = ToyDataset()
    checker = SimpleDuplicateChecker(dataset)

    metric_result = checker.metric()
    print("Duplicate metric:", metric_result)

    cleaned = checker.remedy(metric_result)
    print("Original size:", len(dataset))
    print("Cleaned size:", len(cleaned["ai_ready_dataset"]))
