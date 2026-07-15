Defining your server agent
==========================

.. note::

    We always welcome you to contribute your increments to ``APPFL`` by creating a pull request.

When you need to have your server agent to have additional functionalities that is not provided by the current ``ServerAgent``, you can create your own server agent by inheriting the ``appfl.agent.ServerAgent``. To make the new functionalities available for the client to call, you need to modify ``appfl.comm.mpi.MPIServerCommunicator._invoke_custom_action`` function (for MPI) or ``appfl.comm.grpc.GRPCServerCommunicator.InvokeCustomAction``, and make a call to your own functionalities with a corresponding action name.

For example, if I created a new server agent that has an additional functionality to do federated inference as below

.. code-block:: python

    class MyServerAgentWithFedInference(ServerAgent):
        def fl_inference(*args, **kwargs):
            pass

Then you can make this new function available for client by calling it in the custom action function in the corresponding communicators.

.. code-block:: python


    # For gRPC communicator
    class GRPCServerCommunicator(GRPCCommunicatorServicer):
        ...
        def InvokeCustomAction(self, request, context):
            ...
            if action == 'set_sample_size':
                ...
            elif action == 'fl_inference':
                ...
                val = self.server_agent.fl_inference()
                ...
            ...

    # For MPI communicator
    class MPIServerCommunicator:
        ...
        def _invoke_custom_action(
            self,
            client_id: int,
            request: MPITaskRequest,
        ) -> Optional[MPITaskResponse]:
            ...
            if action == "set_sample_size":
                ...
            elif action == 'fl_inference':
                ...
                val = self.server_agent.fl_inference()
                ...
            ...
