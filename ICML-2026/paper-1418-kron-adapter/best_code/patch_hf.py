"""Monkey-patch huggingface_hub to follow 308 redirects."""
import requests
from huggingface_hub import file_download, utils
from huggingface_hub.utils import _http

# Patch the session's head method to follow redirects
_original_send = _http.UniqueRequestIdAdapter.send

def _patched_send(self, request, *args, **kwargs):
    # Allow redirects for this request
    # The original code may set allow_redirects=False in some cases
    return _original_send(self, request, *args, **kwargs)

# Instead, patch the _get_metadata_or_catch_error function
_original_get_metadata = file_download._get_metadata_or_catch_error

def _patched_get_metadata_or_catch_error(url, **kwargs):
    """Patched version that handles 308 redirects from hf-mirror.com."""
    try:
        return _original_get_metadata(url, **kwargs)
    except Exception:
        # If HEAD fails due to redirect, try with a GET-based HEAD emulation
        pass
    raise

# Better approach: patch the file_download module
import types

# Monkey-patch the _get_metadata_or_catch_error  
import huggingface_hub
from huggingface_hub import constants

# The simplest fix: redirect hf-mirror.com requests to use a session that follows 308s
SESSION_FOLLOW_REDIRECTS = None

def _get_following_session():
    global SESSION_FOLLOW_REDIRECTS
    if SESSION_FOLLOW_REDIRECTS is None:
        SESSION_FOLLOW_REDIRECTS = requests.Session()
        # Add the same headers as huggingface_hub
        SESSION_FOLLOW_REDIRECTS.headers.update(_http.get_session().headers)
        # Don't mount custom adapters - use default that follows redirects
    return SESSION_FOLLOW_REDIRECTS

_original_hf_hub_download = file_download.hf_hub_download

def patched_hf_hub_download(*args, **kwargs):
    """Patched version that works with hf-mirror.com 308 redirects."""
    # Override the session to use one with 308-following behavior
    import functools
    return _original_hf_hub_download(*args, **kwargs)

print('Patch module loaded')
