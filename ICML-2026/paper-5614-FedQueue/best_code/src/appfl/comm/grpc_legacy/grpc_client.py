import grpc
import time
import logging
import numpy as np
from .grpc_utils import proto_to_databuffer, construct_tensor_record
from .grpc_communicator_old_pb2 import (
    Header,
    JobRequest,
    TensorRequest,
    TensorRecord,
    WeightRequest,
    LearningResults,
)
from .channel import create_grpc_channel
from .grpc_communicator_old_pb2_grpc import GRPCCommunicatorV0Stub
from appfl.misc.utils import get_appfl_authenticator


class APPFLgRPCClient:
    def __init__(self, client_id, cfg):
        self.logger = logging.getLogger(__name__)
        self.client_id = client_id
        self.max_message_size = cfg.max_message_size
        self.channel = create_grpc_channel(
            server_uri=cfg.uri,
            use_ssl=cfg.use_ssl,
            use_authenticator=cfg.use_authenticator,
            root_certificates=cfg.client.root_certificates,
            authenticator=get_appfl_authenticator(
                cfg.authenticator, cfg.client.authenticator_kwargs
            )
            if cfg.use_authenticator
            else None,
            max_message_size=self.max_message_size,
        )
        grpc.channel_ready_future(self.channel).result(timeout=60)
        self.stub = GRPCCommunicatorV0Stub(self.channel)
        self.header = Header(server_id=1, client_id=self.client_id)
        self.time_get_job = 0.0
        self.time_get_tensor = 0.0
        self.time_send_results = 0.0

    def get_job(self, job_done):
        request = JobRequest(header=self.header, job_done=job_done)
        start = time.time()
        response = self.stub.GetJob(request)
        end = time.time()
        self.time_get_job += end - start
        self.logger.info(
            f"[Client ID: {self.client_id: 03}] Received JobReponse with (server,round,job)=(%d,%d,%d)",
            response.header.server_id,
            response.round_number,
            response.job_todo,
        )
        return response.round_number, response.job_todo

    def get_tensor_record(self, name, round_number):
        request = TensorRequest(
            header=self.header, name=name, round_number=round_number
        )
        self.logger.debug(
            f"[Client ID: {self.client_id: 03}] Requested Tensor record (name,round)=(%s,%d)",
            name,
            round_number,
        )
        start = time.time()
        response = TensorRecord()
        bytes_received = b""
        for bytes in self.stub.GetTensorRecord(request):
            bytes_received += bytes.data_bytes
        response.ParseFromString(bytes_received)

        end = time.time()
        self.logger.debug(
            f"[Client ID: {self.client_id: 03}] Received Tensor record (name,round)=(%s,%d)",
            name,
            round_number,
        )
        if round_number > 1:
            self.time_get_tensor += end - start
        shape = tuple(response.data_shape)
        flat = np.frombuffer(response.data_bytes, dtype=eval(response.data_dtype))
        nparray = np.reshape(flat, newshape=shape, order="C")

        return nparray

    def get_weight(self, training_size):
        request = WeightRequest(header=self.header, size=training_size)
        response = self.stub.GetWeight(request)
        self.logger.debug(
            f"[Client ID: {self.client_id: 03}] Received weight = %e", response.weight
        )
        return response.weight

    def send_learning_results(self, penalty, primal, dual, round_number):
        primal_tensors = [
            construct_tensor_record(k, np.array(v.cpu())) for k, v in primal.items()
        ]
        dual_tensors = [
            construct_tensor_record(k, np.array(v.cpu())) for k, v in dual.items()
        ]
        proto = LearningResults(
            header=self.header,
            round_number=round_number,
            penalty=penalty[self.client_id],
            primal=primal_tensors,
            dual=dual_tensors,
        )

        databuffer = []
        databuffer += proto_to_databuffer(proto, max_message_size=self.max_message_size)
        start = time.time()
        self.stub.SendLearningResults(iter(databuffer))
        end = time.time()
        if round_number > 1:
            self.time_send_results += end - start

    def get_comm_time(self):
        return self.time_get_job + self.time_get_tensor + self.time_send_results
