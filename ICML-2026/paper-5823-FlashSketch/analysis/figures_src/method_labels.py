from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g


_METHOD_DISPLAY_OVERRIDES = {
    g.METHOD_FLASH_SKETCH: r"$\mathbf{FlashSketch}$ $\mathbf{(Ours)}$",
    g.METHOD_FLASH_BLOCK_ROW: r"$\mathbf{FlashBlockRow}$ $\mathbf{(Ours)}$",
    g.METHOD_SJLT_CUSPARSE: "SJLT (cuSPARSE)",
    g.METHOD_SJLT_GRASS_KERNEL: "SJLT (GraSS Kernel)",
    g.METHOD_GRASS: "GraSS",
    g.METHOD_GAUSSIAN_DENSE_CUBLAS: "Dense Gaussian (cuBLAS)",
    g.METHOD_SRHT_FWHT: "Subsampled Fast \nHadamard Transform",
}


def _unslugify(text):
    """Return a camera-ready name by replacing underscores and title-casing."""
    if not text:
        return ""
    return str(text).replace("_", " ").title()


def format_method_labels(method_labels):
    """Return camera-ready display labels for method labels."""
    formatted = []
    for label in method_labels:
        if not label:
            formatted.append(label)
            continue
        if " " in label:
            head, tail = label.split(" ", 1)
            display = _METHOD_DISPLAY_OVERRIDES.get(head, _unslugify(head))
            formatted.append(f"{display}\n{tail}")
        else:
            formatted.append(_METHOD_DISPLAY_OVERRIDES.get(label, _unslugify(label)))
    return formatted
