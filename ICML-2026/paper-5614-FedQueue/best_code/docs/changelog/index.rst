Changelog
=========

appfl v1.10.0
-------------

New Features
~~~~~~~~~~~~

- Bump the version requirements of ``globus-sdk``, ``globus-compute-sdk``, and ``globus-compute-endpoint`` to the latest versions.
- Add a tutorial for creating multi-user Globus Compute endpoints on Cloud at `here <https://appfl.ai/en/latest/tutorials/examples_multi_user_gcep.html>`_.
- Update the tutorial for local testing APPFLx server at `here <https://github.com/APPFL/APPFL/tree/main/src/appfl/service#appfl-as-a-service-appflx>`_.

appfl v1.9.1
------------

Bug Fixes
~~~~~~~~~

- Include support for ``globus-sdk`` version 4.x for Globus-based authentication in gRPC.

appfl v1.9.0
------------

New Features
~~~~~~~~~~~~

- Introduce support for streamed aggregation in gRPC communication, enabling memory-efficient federated learning for large models and datasets. [`Documentation <https://github.com/APPFL/APPFL/tree/main/examples/memory_profiling>`_].
- Add a new example demonstrating federated learning with Graph Neural Networks (GNNs) using PyTorch Geometric from `@Truc Nguyen <https://github.com/trucndt>`_. [`Documentation <https://github.com/APPFL/APPFL/tree/main/examples/resources/config_gc/pyg>`_].
- Integrate additional privacy-preserving mechanisms from `@aash-mohammad <https://github.com/aash-mohammad>`_, including: secure aggregation and differential privacy via Opacus-Privacy.
- Add tutorial notebooks for running APPFL on NERSC systems, available `here <https://github.com/APPFL/APPFL/tree/main/examples/notebook_tutorials/mnist_nersc>`_.
- Add tutorial notebooks for running APPFL on AI-READI datasets, available `here <https://github.com/APPFL/APPFL/tree/main/examples/notebook_tutorials/aireadi_arvo_course>`_.


Bug Fixes
~~~~~~~~~

- Resolve race conditions in the FedCompass implementation.
- Fix issues when running FedCompass with Globus Compute.
- Improve GitHub workflows and CI/CD actions for testing and deployment stability.


appfl v1.8.0
------------

New Features
~~~~~~~~~~~~

- Add support for GA4GH task execution service (TES) to APPFL communicators (experimental).
- Integrate `Fed-SB <https://arxiv.org/pdf/2502.15436/>`_ into APPFL for efficient federated tuning of large foundation models.
- Add tutorials for leveraging APPFL to training grid foundation models on real-world power grid datasets.

appfl v1.7.0
------------

New Features
~~~~~~~~~~~~

- Optimize the memory usage for FL server, FL client, and different communicators (experimental).
- Add example scripts and documentations for scaling test for federated learning using APPFL among multiple nodes and GPUs.
- Fix some minor bugs and typos.


appfl v1.6.1
------------

- Add documentation for CADRE (Customizable Assurance of Data Readiness) modules
- Add tutorials for running APPFL on FLamby datasets, both on AWS and HPCs
- Add tutorials for Globus-based authentication

appfl v1.5.0
------------

New Features
~~~~~~~~~~~~

- Add CADRE module to APPFL for the assurance of AI data readiness before FL training.
- Support running APPFL tutorials on AWS SageMaker.
- Integrate APPFL github workflow with ALCF Gitlab CI for testing on Polaris with GPUs.

appfl v1.4.1
------------

New Features
~~~~~~~~~~~~

- Add a new tutorial for using shared Globus Compute endpoints at `here <https://appfl.ai/en/latest/tutorials/examples_globus_compute.html#creating-shared-globus-compute-endpoint-on-client-machines>`_.
- Add a new tutorial for joining APPFL-hosted international federation at `here <https://appfl.ai/en/latest/tutorials/examples_international.html>`_.
- Support colab notebooks for running APPFL on Google Colab with a colab connector for easy data transmission.

appfl v1.4.0
------------

New Features
~~~~~~~~~~~~

- Add `Ray <https://www.ray.io/>`_ into communicator, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_ray.html>`_. (`#271 <https://github.com/APPFL/APPFL/pull/271>`_)
- Make ``mpi4py`` optional during installation. (`#264 <https://github.com/APPFL/APPFL/issues/264>`_)

Deprecated
~~~~~~~~~~

- Derecate the usage of ``comm_configs.globus_compute_configs`` for AWS S3 configurations, which is replaced by ``comm_configs.s3_configs``.

appfl v1.3.0
------------

New Features
~~~~~~~~~~~~

- Integrate APPFL with MONAI to use MONAI bundles for federated learning, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_monai.html>`_.  (`#251 <https://github.com/APPFL/APPFL/issues/251>`_)
- Add support for Multi-GPU training using PyTorch DDP, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_gpuclusterrun.html#multi-gpu-training>`_.  (`#254 <https://github.com/APPFL/APPFL/issues/254>`_)
- Integrate `ProxyStore <https://docs.proxystore.dev/latest/>`_ into Globus Compute and gRPC communication protocols for data transmission, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_globus_compute.html#extra-integration-with-proxystore>`_.  (`#252 <https://github.com/APPFL/APPFL/issues/252>`_, `#259 <https://github.com/APPFL/APPFL/issues/259>`_)
- Add three colab-based tutorials at `here <https://appfl.ai/en/latest/notebooks/index.html#colab-notebooks>`_ for running APPFL on Google Colab (`#255 <https://github.com/APPFL/APPFL/issues/255>`_).

appfl v1.2.1
------------

New Features
~~~~~~~~~~~~

- Remove redundant experiment configurations. (`#228 <https://github.com/APPFL/APPFL/issues/228>`_)
- Enhance safety for Globus Compute by only sending a trigger function. (`#227 <https://github.com/APPFL/APPFL/issues/227>`_)

appfl v1.2.0
------------

New Features
~~~~~~~~~~~~

- Improve client name display for running FL experiments by specifying ``client_id`` in the client configuration file.
- Add documentation for using APPFL on ALCF Polaris at `here <https://appfl.ai/en/latest/tutorials/examples_gpuclusterrun.html#grpc-simulation-on-polaris-cluster>`_.
- Allow users to send payload of arbitrary size for custom actions in gRPC communication.
- Add more tests for FL experiments under different scenarios: serial, MPI, batched MPI, and gRPC.
- Integrate ``wandb`` for logging training metadata such as training and validation losses into client trainer, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_wandb.html>`_.

Bug Fixes
~~~~~~~~~

- Fix path issues when running APPFL on Windows.
- Fix batched MPI issue with compression.
- Fix some other small bugs and bump the version of few dependencies.


appfl v1.1.0
------------

New Features
~~~~~~~~~~~~

- Support batched MPI, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_batched_mpi.html>`_.
- Add more data readiness metrics such as PCA plot in this `pull request <https://github.com/APPFL/APPFL/pull/208>`_.
- Backend support for `service.appfl.ai <https://appflx.link/>`_.
- Add documentation for service.appfl.ai at `here <https://appfl.ai/en/latest/tutorials/appflx/index.html>`_.
- Add logging capabilities to the server side to log the training metadata such as the training and validation losses.
- Change documentation theme to ``furo``.

appfl v1.0.5
------------

New Features
~~~~~~~~~~~~

- Add the feature to generate data readiness reports on all client data.
- Update the documentation for adding custom action at `here <https://appfl.ai/en/latest/tutorials/examples_custom_action.html>`_.

appfl v1.0.4
------------

New Features
~~~~~~~~~~~~

- Add documentation for using APPFL with Globus Compute for secure distributed training at `here <https://appfl.ai/en/latest/tutorials/examples_globus_compute.html>`_.

Bug Fixes
~~~~~~~~~

- Fix an issue with Globus Compute at this `commit <https://github.com/APPFL/APPFL/commit/705b5af64389c77e1c0f9f21d1d86c0cc33cd067>`_.

appfl v1.0.3
------------

New Features
~~~~~~~~~~~~

- Add trackback information to the gRPC server to help debug the server-side errors.
- Add a video tutorials for `installing APPFL on AWS <https://youtu.be/ihPofoQwUMs>`_, `creating SSL-encrypted gRPC server <https://youtu.be/3n8a026VqdQ>`_, and `using APPFL to finetune a ViT <https://youtu.be/m4rdOub2Y_o>`_.

Bug Fixes
~~~~~~~~~

- Handle corner cases for server aggregators when the keys in client local models are not consistent with the global model keys.

appfl v1.0.2
------------

New Features
~~~~~~~~~~~~

- Add a new command line interface (CLI), `appfl-setup-ssl` to create necessary certificates for creating SSL-secured gRPC connections between the server and clients.
- Add a tutorial on how to use the CLI, `appfl-setup-ssl`, to create certificates for the server and clients, and enable SSL-secured gRPC connections between the server and clients.
- Add a detailed step-by-step tutorial on how to define custom action with an example to generate a data readiness report on all client data at `here <https://appfl.ai/en/latest/tutorials/examples_custom_action.html>`_.
- Add a APPFL `YouTube channel <https://www.youtube.com/channel/UCzwiJboiJW3dLI0UndnDy5g>`_ to provide video tutorials on how to use APPFL for federated learning research in the future.

Bug Fixes
~~~~~~~~~

- Fix the `issue <https://github.com/APPFL/APPFL/issues/197>`_ regarding client gradient clipping. The clipping is now applied before weights update.

appfl v1.0.1
------------

New Features
~~~~~~~~~~~~

- For the aggregators, the model architecture is set to be an optional initialization parameter, and the aggregators only aggregate the parameters sent by the clients instead of the whole set of model parameters. This is useful when doing federated fine-tuning or federated transfer learning where only part of model parameters are updated / the model architecture is unknown to the aggregator.
- Support easy integration of custom trainer/aggregator: user only needs to provide the custom trainer/aggregator class name and the path to the definition file in the configuration file to use it, instead of modifying the source code.
- Add a detailed step-by-step tutorial on how to use ``APPFL`` to fine-tune a ViT model with a custom trainer.

appfl v1.0.0
------------

Version 1.0.0 of appfl is a major release that refactors the entire codebase to make it more modular, extensible, and functional, while remains backward compatibility with the previous version. The release also included the following changes:

New Features
~~~~~~~~~~~~

- Define server and client agents to act on behalf of the FL server and clients to conduct FL experiments.
- Simplify the configuration process for launching FL experiments by only providing a single YAML file for the server and a YAML file for each client.
- Rebuild the communicator module, supporting MPI, gRPC, and Globus Compute, to robustly exchange model parameters as well as task metadata between the server and clients in both synchronous and asynchronous FL experiment settings.
- Implement Globus-based authentication for secure distributed training with gRPC and Globus Compute - only members within the same specific Globus group can participate in the FL experiment.
- Integrate several lossy and error-bounded lossless compressors to the communicator module for efficient model compression.
- Add documentation for the new version available at `appfl.ai <https://appfl.ai>`_

Deprecated
~~~~~~~~~~

- The previous version of appfl is still seamlessly supported but deprecated and no longer maintained. Users are encouraged to upgrade to the new version for better performance, functionality, and extensibility.
- Examples and tutorials for the previous version are still available in the ``examples/examples_legacy`` directory of the Github appfl repository.
