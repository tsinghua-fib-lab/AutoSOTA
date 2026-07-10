"""
Per-request JiSi state isolation built on contextvars.

Shared state, such as question_bank, embedding_bank, and train_data, remains
owned by the runner. Request-specific state lives in this context so concurrent
requests do not leak model responses, timing data, or precomputed embeddings.
"""
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

_jisi_request_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "jisi_request_ctx", default=None
)


def create_request_context(
    model_raw_response: Optional[Dict[str, Any]] = None,
    test_data: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Create a new request context."""
    return {
        "model_raw_response": dict(model_raw_response) if model_raw_response else {},
        "model_elapsed_time": {},
        "_last_current_response_embed": None,
        "test_data": test_data,
    }


def set_request_context(ctx: Dict[str, Any]):
    """Set the current request context and return a token for reset."""
    return _jisi_request_ctx.set(ctx)


def reset_request_context(token) -> None:
    """Reset the request context."""
    _jisi_request_ctx.reset(token)


def get_request_context() -> Optional[Dict[str, Any]]:
    """Return the current request context, or None when unset."""
    return _jisi_request_ctx.get()
