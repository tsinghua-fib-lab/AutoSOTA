from __future__ import annotations

def preload_cuda_runtime() -> None:
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib
    except Exception:
        return

    for libname in ("cudart", "cudss"):
        try:
            load_nvidia_dynamic_lib(libname)
        except Exception:
            pass
