from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor
EMPTY: MessageStatus
INIT: Job
OK: MessageStatus
QUIT: Job
TRAIN: Job
WEIGHT: Job

class Acknowledgment(_message.Message):
    __slots__ = ["header", "status"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    header: Header
    status: MessageStatus
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        status: _Optional[_Union[MessageStatus, str]] = ...,
    ) -> None: ...

class DataBufferV0(_message.Message):
    __slots__ = ["data_bytes", "size"]
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    size: int
    def __init__(
        self, size: _Optional[int] = ..., data_bytes: _Optional[bytes] = ...
    ) -> None: ...

class Header(_message.Message):
    __slots__ = ["client_id", "server_id"]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: int
    server_id: int
    def __init__(
        self, server_id: _Optional[int] = ..., client_id: _Optional[int] = ...
    ) -> None: ...

class JobRequest(_message.Message):
    __slots__ = ["header", "job_done"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    JOB_DONE_FIELD_NUMBER: _ClassVar[int]
    header: Header
    job_done: Job
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        job_done: _Optional[_Union[Job, str]] = ...,
    ) -> None: ...

class JobResponse(_message.Message):
    __slots__ = ["header", "job_todo", "round_number"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    JOB_TODO_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    header: Header
    job_todo: Job
    round_number: int
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        round_number: _Optional[int] = ...,
        job_todo: _Optional[_Union[Job, str]] = ...,
    ) -> None: ...

class LearningResults(_message.Message):
    __slots__ = ["dual", "header", "penalty", "primal", "round_number"]
    DUAL_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRIMAL_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    dual: _containers.RepeatedCompositeFieldContainer[TensorRecord]
    header: Header
    penalty: float
    primal: _containers.RepeatedCompositeFieldContainer[TensorRecord]
    round_number: int
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        round_number: _Optional[int] = ...,
        penalty: _Optional[float] = ...,
        primal: _Optional[_Iterable[_Union[TensorRecord, _Mapping]]] = ...,
        dual: _Optional[_Iterable[_Union[TensorRecord, _Mapping]]] = ...,
    ) -> None: ...

class TensorRecord(_message.Message):
    __slots__ = ["data_bytes", "data_dtype", "data_shape", "name"]
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    DATA_DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_SHAPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    data_dtype: str
    data_shape: _containers.RepeatedScalarFieldContainer[int]
    name: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        data_shape: _Optional[_Iterable[int]] = ...,
        data_bytes: _Optional[bytes] = ...,
        data_dtype: _Optional[str] = ...,
    ) -> None: ...

class TensorRequest(_message.Message):
    __slots__ = ["header", "name", "round_number"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    header: Header
    name: str
    round_number: int
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        name: _Optional[str] = ...,
        round_number: _Optional[int] = ...,
    ) -> None: ...

class WeightRequest(_message.Message):
    __slots__ = ["header", "size"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    header: Header
    size: int
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        size: _Optional[int] = ...,
    ) -> None: ...

class WeightResponse(_message.Message):
    __slots__ = ["header", "weight"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    header: Header
    weight: float
    def __init__(
        self,
        header: _Optional[_Union[Header, _Mapping]] = ...,
        weight: _Optional[float] = ...,
    ) -> None: ...

class Job(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class MessageStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
