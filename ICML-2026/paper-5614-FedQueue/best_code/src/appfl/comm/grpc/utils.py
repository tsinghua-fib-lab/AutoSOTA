import io
import torch
from .grpc_communicator_pb2 import DataBuffer


def proto_to_databuffer(proto, max_message_size=(2 * 1024 * 1024)):
    max_message_size = int(0.9 * max_message_size)
    data_bytes = proto.SerializeToString()
    data_bytes_size = len(data_bytes)
    message_size = (
        data_bytes_size if max_message_size > data_bytes_size else max_message_size
    )

    for i in range(0, data_bytes_size, message_size):
        chunk = data_bytes[i : i + message_size]
        msg = DataBuffer(data_bytes=chunk)
        yield msg


def serialize_model(model):
    """Serialize a model to a byte string."""
    buffer = io.BytesIO()
    torch.save(model, buffer)
    return buffer.getvalue()


def deserialize_model(model_bytes):
    """Deserialize a model from a byte string."""
    return torch.load(io.BytesIO(model_bytes))


def load_credential_from_file(filepath):
    with open(filepath, "rb") as f:
        return f.read()
