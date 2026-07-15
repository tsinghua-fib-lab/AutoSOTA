from enum import Enum
from dataclasses import dataclass


class MPITask(Enum):
    """MPI task type"""

    GET_CONFIGURATION = 0
    GET_GLOBAL_MODEL = 1
    UPDATE_GLOBAL_MODEL = 2
    INVOKE_CUSTOM_ACTION = 3


class MPIServerStatus(Enum):
    """MPI server status"""

    RUN = 0
    DONE = 1
    ERROR = 2


@dataclass
class MPITaskRequest:
    """MPI task request"""

    payload: bytes = b""
    meta_data: str = ""


@dataclass
class MPITaskResponse:
    """MPI task response"""

    status: int = MPIServerStatus.RUN.value
    payload: bytes = b""
    meta_data: str = ""
