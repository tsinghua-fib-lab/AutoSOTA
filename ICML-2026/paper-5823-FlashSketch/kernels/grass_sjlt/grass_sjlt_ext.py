from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from kernels.so_loader import load_extension


_EXTENSION_NAME = "grass_sjlt_ext"
_EXTENSION = None


def _get_extension():
    """Return the loaded extension, raising if missing."""
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load_extension(_EXTENSION_NAME)
    return _EXTENSION


def sjlt_projection_cuda(
    input_tensor,
    rand_indices,
    rand_signs,
    proj_dim,
    c,
    threads,
    fixed_blocks,
):
    """Apply the GraSS SJLT projection kernel."""
    ext = _get_extension()
    return ext.sjlt_projection_cuda(
        input_tensor,
        rand_indices,
        rand_signs,
        int(proj_dim),
        int(c),
        int(threads),
        int(fixed_blocks),
    )


def sjlt_transpose_cuda(
    input_tensor,
    rand_indices,
    rand_signs,
    original_dim,
    c,
    threads,
    fixed_blocks,
):
    """Apply the GraSS SJLT transpose kernel."""
    ext = _get_extension()
    return ext.sjlt_transpose_cuda(
        input_tensor,
        rand_indices,
        rand_signs,
        int(original_dim),
        int(c),
        int(threads),
        int(fixed_blocks),
    )
