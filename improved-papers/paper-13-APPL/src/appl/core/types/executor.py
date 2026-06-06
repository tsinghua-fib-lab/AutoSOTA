from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from typing import Optional, Union


class ExecutorNotFoundError(Exception):
    """Custom exception to indicate the executor is not found."""

    def __init__(self, message="Executor not found"):
        """Initialize the exception."""
        super().__init__(message)


class ExecutorType(str, Enum):
    """The type of the executor."""

    LLM_THREAD_POOL = "llm_thread_pool"
    GENERAL_THREAD_POOL = "general_thread_pool"
    GENERAL_PROCESS_POOL = "general_process_pool"
    NEW_THREAD = "new_thread"
    NEW_PROCESS = "new_process"


def is_thread_executor(executor_type: ExecutorType) -> bool:
    """Check if the executor is a thread executor."""
    return "thread" in executor_type.value


class GlobalExecutors:
    """The global executors."""

    _llm_thread_executor: Optional[ThreadPoolExecutor] = None
    _general_thread_executor: Optional[ThreadPoolExecutor] = None
    _general_process_executor: Optional[ProcessPoolExecutor] = None

    def set_executor(
        self,
        executor_type: ExecutorType,
        executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]],
    ) -> None:
        """Set the global executors."""
        if executor_type == ExecutorType.LLM_THREAD_POOL:
            if not isinstance(executor, ThreadPoolExecutor):
                raise ValueError("LLM thread executor must be a ThreadPoolExecutor")
            self._llm_thread_executor = executor
        elif executor_type == ExecutorType.GENERAL_THREAD_POOL:
            if not isinstance(executor, ThreadPoolExecutor):
                raise ValueError("General thread executor must be a ThreadPoolExecutor")
            self._general_thread_executor = executor
        elif executor_type == ExecutorType.GENERAL_PROCESS_POOL:
            if not isinstance(executor, ProcessPoolExecutor):
                raise ValueError(
                    "General process executor must be a ProcessPoolExecutor"
                )
            self._general_process_executor = executor
        else:
            raise ValueError(f"Cannot set executor of type: {executor_type}")

    def get_executor(
        self, executor_type: ExecutorType
    ) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Get the executor of a given type."""
        if executor_type == ExecutorType.LLM_THREAD_POOL:
            if self._llm_thread_executor is None:
                raise ExecutorNotFoundError("LLM thread executor is not set")
            return self._llm_thread_executor
        elif executor_type == ExecutorType.GENERAL_THREAD_POOL:
            if self._general_thread_executor is None:
                raise ExecutorNotFoundError("General thread executor is not set")
            return self._general_thread_executor
        elif executor_type == ExecutorType.GENERAL_PROCESS_POOL:
            if self._general_process_executor is None:
                raise ExecutorNotFoundError("General process executor is not set")
            return self._general_process_executor
        else:
            return self._create_executor(executor_type)

    def _create_executor(
        self, executor_type: ExecutorType
    ) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Create an executor of a given type."""
        if executor_type == ExecutorType.NEW_THREAD:
            return ThreadPoolExecutor(max_workers=1)
        elif executor_type == ExecutorType.NEW_PROCESS:
            return ProcessPoolExecutor(max_workers=1)
        else:
            raise ValueError(f"Invalid executor type: {executor_type}")


global_executors = GlobalExecutors()
