Example: Add a Custom Action
============================

In ``APPFL``, the server supports several actions such as getting general client configurations, getting the global model parameters, and updating the global model parameters (i.e., federated training). However, in some cases, you may want to add a custom action to the server, such as fedearated evaluation. In this example, we show how to add a custom action to the server to generate a data readiness report for all the clients local datasets.

.. _client-side-implementation:

Client Side Implementation
--------------------------

In this example, we focus on the client-driven communication pattern (MPI and gRPC), where the clients sends requests to the server for any actions they want to perform. In this case, the client-side can simply define a function to generate the data readiness report for its local dataset, and then send a request to the server to generate the report for all the clients. The server handles the action synchronously, meaning it waits to receive requests from all clients before proceeding with the generation of the aggregated readiness report.

However, as the ``APPFL`` defines client agent ``appfl.agent.ClientAgent`` to act on behalf of the client, we highly recommend to define the custom action within the client agent either by extending the ``appfl.agent.ClientAgent`` or by adding a new method to the existing ``appfl.agent.ClientAgent``. In this example, we create a new method for the existing client to generate the data readiness report.

.. note::
    If you think your custom action is useful for the community, please consider define it within the ``appfl.client.ClientAgent`` directly and contribute it to the ``APPFL`` framework by creating a pull request.

.. _new-method-client-agent:

New Method in Client Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``ClientAgent``, we define a new function ``generate_readiness_report`` to generate the data readiness report for the local dataset. The function will be called by the client to generate the readiness report for its local dataset. The function returns a dictionary containing the readiness metrics and plots. Below is the implementation of the function ``generate_readiness_report`` in the client agent.

.. literalinclude:: ../../src/appfl/agent/client.py
    :start-after: torch.save(self.model.state_dict(), checkpoint_path)
    :end-before: def _create_logger
    :language: python
    :caption: generate_readiness_report - A function in ``ClientAgent`` which can generate data readiness evaluations and plots for the dataset.

The corresponding computation functions for the metrics and plots are defined in the ``src.appfl.misc.data_readiness.metrics.py`` and ``src.appfl.misc.data_readiness.plots.py`` files respectively.

.. _send-request-to-server:

Send Request to the Server
~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether it is MPI or gRPC communicator, ``APPFL`` provides an interface function, ``invoke_custom_action``, for user to send any custom request to the server, as shown below. The handler for the custom action should be defined on the server side, as shown in the next section.

.. note::

    See the sections :ref:`launch experiments` for details on how to create the communicator.


.. code-block:: python

    client_communicator = ... # this can be either MPI or gRPC communicator, see the following sections
    data_readiness = client_agent.generate_readiness_report(client_config)
    client_communicator.invoke_custom_action(action='get_data_readiness_report', **data_readiness)

.. _server-side-implementation:

Server Side Implementation
--------------------------

On the server side, the server should update a handler for processing the custom action request from the client. In this example, we will show how to update the custom action handler for both the MPI and gRPC communicator.

.. _mpi-communicator:

MPI Communicator
~~~~~~~~~~~~~~~~

User needs to update the ``APPFL`` source code's MPI server communicator at ``appfl.comm.mpi.mpi_server_communicator.MPIServerCommunicator``. You need to update its ``_invoke_custom_action`` method to handle the custom action request from the client. Here is an example implementation for action ``get_data_readiness_report`` which synchronously waits for all clients to send their readiness evaluations and metadata. Once all clients have submitted their data, it aggregates the results and sends them to the server agent to generate and output the readiness report.

.. code-block:: python

    def _invoke_custom_action(
        self,
        client_id: int,
        request: MPITaskRequest,
    ) -> Optional[MPITaskResponse]:
        ...
        if action == "set_sample_size":
            ...
        elif action == "close_connection":
            ...
        elif action == "get_data_readiness_report":
            num_clients = self.server_agent.get_num_clients()
            if not hasattr(self, "_dr_metrics_lock"):
                self._dr_metrics = {}
                self._dr_metrics_client_ids = set()
                self._dr_metrics_lock = threading.Lock()
            with self._dr_metrics_lock:
                self._dr_metrics_client_ids.add(client_id)
                for k, v in meta_data.items():
                    if k not in self._dr_metrics:
                        self._dr_metrics[k] = {}
                    self._dr_metrics[k][client_id] = v
                if len(self._dr_metrics_client_ids) == num_clients:
                    self.server_agent.data_readiness_report(self._dr_metrics)
                    response = MPITaskResponse(
                        status=MPIServerStatus.RUN.value,
                    )
                    response_bytes = response_to_byte(response)
                    for client_id in self._dr_metrics_client_ids:
                        self.comm.Send(response_bytes, dest=client_id, tag=client_id)
                    self._dr_metrics = {}
                    self._dr_metrics_client_ids = set()
            return None
        else:
            raise NotImplementedError(f"Custom action {action} is not implemented.")

.. _grpc-communicator:

gRPC Communicator
~~~~~~~~~~~~~~~~~

Similar to the MPI communicator, user needs to update the ``APPFL`` source code's gRPC server communicator at ``appfl.comm.grpc.grpc_server_communicator.GRPCServerCommunicator``. You need to update its ``InvokeCustomAction`` method to handle the custom action request from the client. Here is an example implementation for action ``get_data_readiness_report`` which synchronously waits for all clients to send their readiness evaluations and metadata. Once all clients have submitted their data, it aggregates the results and sends them to the server to generate and output the readiness report.

.. code-block:: python

    def InvokeCustomAction(self, request, context):
        action = request.action
        meta_data = yaml.safe_load(request.meta_data) if len(request.meta_data) > 0 else {}
        if action == "set_sample_size":
            ...
        elif action == "close_connection":
            ...
        elif action == "get_data_readiness_report":
                num_clients = self.server_agent.get_num_clients()
                if not hasattr(self, "_dr_metrics_lock"):
                    self._dr_metrics = {}
                    self._dr_metrics_futures = {}
                    self._dr_metrics_lock = threading.Lock()
                with self._dr_metrics_lock:
                    for k, v in meta_data.items():
                        if k not in self._dr_metrics:
                            self._dr_metrics[k] = {}
                        self._dr_metrics[k][client_id] = v
                    _dr_metric_future = Future()
                    self._dr_metrics_futures[client_id] = _dr_metric_future
                    if len(self._dr_metrics_futures) == num_clients:
                        self.server_agent.data_readiness_report(self._dr_metrics)
                        for client_id, future in self._dr_metrics_futures.items():
                            future.set_result(None)
                        self._dr_metrics = {}
                        self._dr_metrics_futures = {}
                # waiting for the data readiness report to be generated for synchronization
                _dr_metric_future.result()
                response = CustomActionResponse(
                    header=ServerHeader(status=ServerStatus.DONE),
                )
                return response
            else:
                raise NotImplementedError(f"Custom action {action} is not implemented.")


.. _server-agent-report-generation:

Server Agent Report Generation
------------------------------
The ``ServerAgent`` should have a method to generate the readiness report for all the clients. The method should take the readiness evaluations and metadata from all the clients and generate the aggregated readiness report. Here the server generates an HTML and JSON report and outputs it to ``.output`` directory. Shown below is the implementation of the method ``data_readiness_report`` in the ``ServerAgent``.

.. literalinclude:: ../../src/appfl/agent/server.py
    :start-after: self.closed_clients.add(client_id)
    :end-before: def server_terminated(self):
    :language: python
    :caption: data_readiness_report - A method in the ``ServerAgent`` that generates a single aggregated readiness report, outputting the results in both HTML and JSON formats for all clients.

The helper functions for generating the HTML and JSON readiness report are defined in the ``src.appfl.misc.data_readiness.report.py`` file.

.. _launch-experiments:

Launch Experiments
------------------

Data readiness report generation is integrated into the standard workflow. The report will be generated before the local training and global model update iterations. To generate the report, you must set ``generate_dr_report: True`` in the server configuration, along with specifying which metrics and plots to include in the evaluation. These settings should be defined in the ``client_configs`` section of the server configuration file. Below is an example of how to set the configurations for the data readiness report generation.

.. code-block:: yaml

    client_configs:
        ...
        data_readiness_configs:
            generate_dr_report: True                    # Enable or disable the generation of data readiness report
            output_dirname: "./output"                  # Directory to save the report
            output_filename: "data_readiness_report"    # Name of the report file
            dr_metrics:                                 # Metrics to evaluate data readiness
                class_imbalance: True                   # Check for class imbalance degree
                sample_size: True                       # Evaluate the sample size
                ...
            plot:                                       # Plots to include in the report
                class_distribution_plot: True           # Generate a class distribution plot
                ...

.. _mpi-experiment:

MPI Experiment
~~~~~~~~~~~~~~

Once the configurations are set, you can run the experiment using the following command while in the ``examples`` directory.

.. code-block:: bash

    mpiexec -n 6 python mpi/run_mpi.py

.. _grpc-experiment:

gRPC Experiment
~~~~~~~~~~~~~~~

Once the configurations are set, you can run the experiment while in the ``examples`` directory. To run the experiment, you need to start the server first in one terminal, and then start two clients in two separate terminals. You can run the server using the following command:

.. code-block:: bash

    python grpc/run_server.py

Then, you can run the clients using the following command:

.. code-block:: bash

    python grpc/run_client.py --config resources/configs/mnist/client_1.yaml
    python grpc/run_client.py --config resources/configs/mnist/client_2.yaml # run this in a separate terminal

The server will generate the data readiness report for all the clients and save it in the ``output`` directory with the name as specified in the configuration file.

.. code-block:: bash

    examples
    |--- output
         |--- data_readiness_report.html
         |--- data_readiness_report.json
