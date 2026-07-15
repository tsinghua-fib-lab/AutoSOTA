Model Compression
=================

User can compress the model parameters using the following lossy compression techniques:

- `SZ2: Error-bounded Lossy Compressor for HPC Data <https://github.com/szcompressor/SZ>`_
- `SZ3: A Modular Error-bounded Lossy Compression Framework for Scientific Datasets <https://github.com/szcompressor/SZ3>`_
- `SZx: An Ultra-fast Error-bounded Lossy Compressor for Scientific Datasets <https://github.com/szcompressor/SZx>`_
- `ZFP: Compressed Floating-Point and Integer Arrays <https://pypi.org/project/zfpy/>`_

Installation
------------

User can install the compressors by running the following command:

.. code-block:: bash

    appfl-install-compressor

.. note::
    SZx is not open source so we omit its installation here. Please install it manually by contacting the author.

Configuration
-------------
User can create a compressor instance by providing a `Config` object and setting necessary parameters. The following example shows how to create a `SZ2` compressor instance:

.. code-block:: python

    from appfl.config import Config
    from appfl.compressor import Compressor

    # set the configuration
    config = Config()
    config.enable_compression = True
    config.lossy_compressor = "SZ2"         # ["SZ2", "SZ3", "ZFP", "SZx"]
    config.lossless_compressor = "blosc"    # ["blosc", "zstd", "gzip", "zlib", "lzma"]
    config.error_bounding_mode = "REL"      # ["ABS", "REL"]
    config.error_bound = 1e-3

    # create the compressor instance
    compressor = Compressor(config)

Usage
-----
User can compress the model parameters by calling the `compress_model` method of the `Compressor` instance. To decompress the model parameters, use can call the `decompress_model` method by providing the compressed model parameters and a model instance (as a reference to the model architecture).

.. code-block:: python

    # create a CNN model
    cnn = CNN(num_channel=3, num_classes=10, nun_pixel=32)

    # compress the model parameters
    compressed_model, lossy_elements = compressor.compress_model(cnn.state_dict())

    # decompress the model parameters
    decompressed_model = compressor.decompress_model(compressed_model, model)

.. note::
    The CNN model used in the above example is defined as follows:

    .. literalinclude:: /../examples/models/cnn.py
        :language: python
        :caption: User-defined convolutionsl neural network model: examples/models/cnn.py
