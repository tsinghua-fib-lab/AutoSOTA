from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g

METHOD_COLOR_MAP = {
    g.METHOD_GAUSSIAN_DENSE_CUBLAS: "#1f77b4",
    g.METHOD_FLASH_BLOCK_ROW: "#ff7f0e",
    g.METHOD_FLASH_SKETCH: "#2ca02c",
    g.METHOD_SJLT_CUSPARSE: "#d62728",
    g.METHOD_SRHT_FWHT: "#5254a3",
    g.METHOD_SJLT_GRASS_KERNEL: "#000000",
    g.METHOD_GRASS: "#000000",
}


def get_method_color_map(methods):
    """Return a stable sketch-method->color mapping."""
    missing = sorted({method for method in methods if method not in METHOD_COLOR_MAP})
    if missing:
        raise ValueError(f"Missing colors for methods: {', '.join(missing)}")
    return {method: METHOD_COLOR_MAP[method] for method in methods}
