
import huggingface_hub.file_download as _fdp
_or = _fdp._request_wrapper
def _pr(**kw):
    kw[allow_redirects] = True
    return _or(**kw)
_fdp._request_wrapper = _pr

