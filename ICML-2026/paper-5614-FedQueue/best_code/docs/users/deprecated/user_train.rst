Training PPFL
=============

To run PPFL with decentralized data on multiple machines, we use `gRPC <https://grpc.io/docs/>`_ that  allows clients from different platforms to seamlessly connect to the server for federated learning.
This contrasts with MPI where all clients and servers should reside in the same cluster.

gRPC uses the HTTP/2 protocol.
A server hosts a service specified by a URI (e.g., ``moonshot.cels.anl.gov:50051`` where ``50051`` is the port number) for communication and clients send requests and receive responses via that URI. Communication protocols between a server and clients are defined via `Protocol Buffers <https://developers.google.com/protocol-buffers/docs/overview>`_, which are defined in the ``appfl/protos`` directory.
For more details, we refer to `gRPC <https://grpc.io/docs/>`_.

The API functions to run gRPC are defined as follows:

.. autofunction:: appfl.run_grpc_server.run_server

.. autofunction:: appfl.run_grpc_client.run_client
