APPFL Compressor
================

Currently, APPFL supports the following lossy compressors:

- `SZ2: Error-bounded Lossy Compressor for HPC Data <https://github.com/szcompressor/SZ>`_ via ``SZ2Compressor``
- `SZ3: A Modular Error-bounded Lossy Compression Framework for Scientific Datasets <https://github.com/szcompressor/SZ3>`_ via ``SZ3Compressor``
- `SZx: An Ultra-fast Error-bounded Lossy Compressor for Scientific Datasets <https://github.com/szcompressor/SZx>`_ via ``SZxCompressor``
- `ZFP: Compressed Floating-Point and Integer Arrays <https://pypi.org/project/zfpy/>`_ via ``ZFPCompressor``

Installation
------------

Users need to first make sure that they have installed the required packages for the compressors when installing `appfl`.

.. code-block:: bash

    pip install appfl
    # OR: If installed from source code
    pip install -e .

Users then can install the compressors by running the following command:

.. code-block:: bash

    appfl-install-compressor

.. note::
    SZx is not open source so we omit its installation here. Please install it manually by contacting the author.

Functionalities
---------------

The APPFL compressor can be used to compress and decompress the model parameters by invoking the ``compressor.compress_model`` and ``compressor.decompress_model`` methods. For example, for ``SZ2Compressor``, the following is the method signature:

.. code-block:: python

    class SZ2Compressor:
        def __init__(self, compressor_config: DictConfig):
            pass

        def compress_model(
            self,
            model: Union[dict, OrderedDict, List[Union[dict, OrderedDict]]],
            batched: bool=False
        ) -> bytes:
            """
            Compress all the parameters of local model(s) for efficient communication. The local model can be batched as a list.
            :param model: local model parameters (can be nested)
            :param batched: whether the input is a batch of models
            :return: compressed model parameters as bytes
            """
            pass

        def decompress_model(
            self,
            compressed_model: bytes,
            model: Union[dict, OrderedDict],
            batched: bool=False
        )-> Union[OrderedDict, dict, List[Union[OrderedDict, dict]]]:
            """
            Decompress all the communicated model parameters. The local model can be batched as a list.
            :param compressed_model: compressed model parameters as bytes
            :param model: a model sample for de-compression reference
            :param batched: whether the input is a batch of models
            :return decompressed_model: decompressed model parameters
            """

Configuration
-------------
User can configure the compressor by setting it ``client_configs.comm_configs.compressor_configs`` in the server configuration file. The following is an example of the configuration:

.. code-block:: python

    client_configs:
        comm_configs:
            compressor_configs:
            enable_compression: True
            lossy_compressor:  "SZ2Compressor"
            lossless_compressor: "blosc"
            error_bounding_mode: "REL"
            error_bound: 1e-3
            param_cutoff: 1024

Usage in APPFL
--------------

The compressor is used in the ``ClientAgent.get_parameters`` method to compress the model parameters using ``compressor.compress_model`` before sending them to the server, as shown below

.. code-block:: python

    def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
        """Return parameters for communication"""
        params = self.trainer.get_parameters()
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)
        return params if metadata is None else (params, metadata)

On the server side, the model parameters are decompressed using ``compressor.decompress_model`` before updating the model by the ``ServerAgent.global_update``.


Stand-alone Usage
-----------------

In APPFL, the compressor is seamlessly integrated into the communication process for user's convenience. However, users can also use the compressor as a stand-alone tool. The following is an example of how to use the compressor to compress and decompress the model parameters.

.. code-block:: python

    from torch import nn
    from omegaconf import OmegaConf
    from appfl.compressor import SZ2Compressor

    # Define a test model
    model = nn.Sequential(
        nn.Conv2d(1, 20, 5),
        nn.ReLU(),
        nn.Conv2d(20, 64, 5),
        nn.ReLU()
    )

    # Load the compressor configuration
    compressor_config = OmegaConf.create({
        "lossless_compressor": "blosc",
        "error_bounding_mode": "REL",
        "error_bound": 1e-3,
        "param_cutoff": 1024
    })

    # Initialize the compressor
    compressor = SZ2Compressor(compressor_config)

    # Compress the model parameters
    compressed_model = compressor.compress_model(model.state_dict())

    # Decompress the model parameters
    decompressed_model = compressor.decompress_model(compressed_model, model)
