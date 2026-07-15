from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class ServerStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN: _ClassVar[ServerStatus]
    DONE: _ClassVar[ServerStatus]
    ERROR: _ClassVar[ServerStatus]

RUN: ServerStatus
DONE: ServerStatus
ERROR: ServerStatus

class DataBuffer(_message.Message):
    __slots__ = ("data_bytes",)
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    def __init__(self, data_bytes: _Optional[bytes] = ...) -> None: ...

class ClientHeader(_message.Message):
    __slots__ = ("client_id",)
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    def __init__(self, client_id: _Optional[str] = ...) -> None: ...

class ServerHeader(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: ServerStatus
    def __init__(self, status: _Optional[_Union[ServerStatus, str]] = ...) -> None: ...

class ConfigurationRequest(_message.Message):
    __slots__ = ("header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    meta_data: str
    def __init__(
        self,
        header: _Optional[_Union[ClientHeader, _Mapping]] = ...,
        meta_data: _Optional[str] = ...,
    ) -> None: ...

class ConfigurationResponse(_message.Message):
    __slots__ = ("header", "configuration")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    configuration: str
    def __init__(
        self,
        header: _Optional[_Union[ServerHeader, _Mapping]] = ...,
        configuration: _Optional[str] = ...,
    ) -> None: ...

class GetGlobalModelRequest(_message.Message):
    __slots__ = ("header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    meta_data: str
    def __init__(
        self,
        header: _Optional[_Union[ClientHeader, _Mapping]] = ...,
        meta_data: _Optional[str] = ...,
    ) -> None: ...

class GetGlobalModelRespone(_message.Message):
    __slots__ = ("header", "global_model", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    global_model: bytes
    meta_data: str
    def __init__(
        self,
        header: _Optional[_Union[ServerHeader, _Mapping]] = ...,
        global_model: _Optional[bytes] = ...,
        meta_data: _Optional[str] = ...,
    ) -> None: ...

class UpdateGlobalModelRequest(_message.Message):
    __slots__ = ("header", "local_model", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    LOCAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    local_model: bytes
    meta_data: str
    def __init__(
        self,
        header: _Optional[_Union[ClientHeader, _Mapping]] = ...,
        local_model: _Optional[bytes] = ...,
        meta_data: _Optional[str] = ...,
    ) -> None: ...

class UpdateGlobalModelResponse(_message.Message):
    __slots__ = ("header", "global_model", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    global_model: bytes
    meta_data: str
    def __init__(
        self,
        header: _Optional[_Union[ServerHeader, _Mapping]] = ...,
        global_model: _Optional[bytes] = ...,
        meta_data: _Optional[str] = ...,
    ) -> None: ...

class CustomActionRequest(_message.Message):
    __slots__ = ("header", "action", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    action: str
    meta_data: str
    def __init__(
        self,
        header: _Optional[_Union[ClientHeader, _Mapping]] = ...,
        action: _Optional[str] = ...,
        meta_data: _Optional[str] = ...,
    ) -> None: ...

class CustomActionResponse(_message.Message):
    __slots__ = ("header", "results")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    results: str
    def __init__(
        self,
        header: _Optional[_Union[ServerHeader, _Mapping]] = ...,
        results: _Optional[str] = ...,
    ) -> None: ...
