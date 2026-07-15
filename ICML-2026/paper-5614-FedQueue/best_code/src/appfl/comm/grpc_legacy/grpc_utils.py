from .grpc_communicator_old_pb2 import DataBufferV0
from .grpc_communicator_old_pb2 import TensorRecord


def construct_tensor_record(name, nparray):
    return TensorRecord(
        name=name,
        data_shape=list(nparray.shape),
        data_bytes=nparray.tobytes(order="C"),
        data_dtype="np." + str(nparray.dtype),
    )


def proto_to_databuffer(proto, max_message_size=(2 * 1024 * 1024)):
    max_message_size = max_message_size - 16  # 16 bytes for the message size field
    data_bytes = proto.SerializeToString()
    data_bytes_size = len(data_bytes)
    message_size = (
        data_bytes_size if max_message_size > data_bytes_size else max_message_size
    )

    for i in range(0, data_bytes_size, message_size):
        chunk = data_bytes[i : i + message_size]
        msg = DataBufferV0(size=message_size, data_bytes=chunk)
        yield msg
