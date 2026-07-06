from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()


def prepare_sketch_input(A, cfg):
    """Return (A_prepped, was_transposed) for sketching."""
    if getattr(cfg, "expects_transposed", False):
        A_prepped = A.transpose(0, 1)
        if not A_prepped.is_contiguous():
            A_prepped = A_prepped.contiguous()
        return A_prepped, True
    if not A.is_contiguous():
        A = A.contiguous()
    return A, False


def finalize_sketch_output(SA, was_transposed):
    """Return sketch output in (k, n) shape without forcing contiguity."""
    if was_transposed:
        return SA.transpose(0, 1)
    return SA
