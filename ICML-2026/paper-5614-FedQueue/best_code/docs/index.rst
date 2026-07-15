.. APPFL documentation master file, created by
   sphinx-quickstart on Fri Oct 29 15:16:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
APPFL
=====

APPFL, Advanced Privacy-Preserving Federated Learning, is an open-source software framework that allows research communities to implement, test, and validate various ideas related to privacy-preserving federated learning (FL), and deploy real FL experiments safely and easily among distributed clients to train ML models.
With this framework, developers and/or users can easily

* Train any user-defined machine learning model on decentralized data with optional differential privacy and client authentication,
* Simulate various synchronous, asynchronous and PPFL algorithms on high-performance computing (HPC) architecture with MPI,
* Implement customizations in a plug-and-play manner for all aspects of FL, including aggregation algorithms, server scheduling strategies, and client local trainers.

Technical Components
====================

APPFL is primarily composed of the following six technical components: *Aggregator*, *Scheduler*, *Trainer*, *Privacy*, *Communicator*, and *Compressor*, all with easy interface for user customizations.

.. grid:: 3

   .. grid-item-card::

      Aggregator
      ^^^^^^^^^^
      Supports several popular algorithms to aggregate one or several client local models.

   .. grid-item-card::

      Scheduler
      ^^^^^^^^^
      Supports several synchronous and asynchronous scheduling algorithms at the server-side to deal with different arrival times of client local models.

   .. grid-item-card::

      Trainer
      ^^^^^^^
      Supports several client local trainers for various training tasks.

.. grid:: 3

   .. grid-item-card::

      Privacy
      ^^^^^^^
      Supports several global/local differential privacy schemes.

   .. grid-item-card::

      Communicator
      ^^^^^^^^^^^^
      Supports MPI for single-machine/cluster simulation, and gRPC and Globus Compute with authenticator for secure distributed training.

   .. grid-item-card::

      Compressor
      ^^^^^^^^^^
      Supports several lossy compressors, including `SZ2 <https://github.com/szcompressor/SZ>`_, `SZ3 <https://github.com/szcompressor/SZ3>`_, `ZFP <https://pypi.org/project/zfpy/>`_, and `SZx <https://github.com/szcompressor/SZx>`_.

APPFL aims to maintain a composable design of the package, where each technical component is extendible and independent from each other. Users can use the configuration files to specify the desired item from each component to launch FL experiments according their needs.

Main Topics
===========

.. toctree::
   :maxdepth: 1

   install/index
   tutorials/index
   users/index
   community/index
   contribution/index
   publication/index
   changelog/index


Citation
========

If you find APPFL useful for your research or development, please cite the following papers:

.. code-block:: latex

   @article{li2024advances,
      title={Advances in APPFL: A Comprehensive and Extensible Federated Learning Framework},
      author={Li, Zilinghan and He, Shilan and Yang, Ze and Ryu, Minseok and Kim, Kibaek and Madduri, Ravi},
      journal={arXiv preprint arXiv:2409.11585},
      year={2024}
   }

   @inproceedings{ryu2022appfl,
      title={APPFL: open-source software framework for privacy-preserving federated learning},
      author={Ryu, Minseok and Kim, Youngdae and Kim, Kibaek and Madduri, Ravi K},
      booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
      pages={1074--1083},
      year={2022},
      organization={IEEE}
   }

Acknowledgement
===============
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
