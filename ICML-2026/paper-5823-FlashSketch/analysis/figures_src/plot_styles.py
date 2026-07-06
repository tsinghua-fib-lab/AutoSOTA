from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g


BLOCK_PERM_METHODS = {
    g.METHOD_FLASH_SKETCH,
}
BLOCK_PERM_PRIMARY = (1, 4)
BLOCK_PERM_LINEWIDTH = {
    (1, 4): 3.0,
}
DEFAULT_ALPHA = 1.0
DEFAULT_LINEWIDTH = 2.0
KAPPA_LINEWIDTH = 2.4
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
VARIANT_FIELDS = (
    "s",
    "kappa",
    "block_rows",
    "block_cols",
    "block_size",
    "tc",
    "tr",
    "br",
    "d_block",
    "s_block",
    "use_fused",
    "return_contiguous",
    "use_csr",
    "scale",
)

def _variant_key(row, fields):
    """Return a tuple key describing a config variant."""
    return tuple(row.get(field) for field in fields)


def build_style_map(grouped, color_map):
    """Return a method_label->style mapping with color, linestyle, alpha, linewidth."""
    if not hasattr(grouped, "columns"):
        raise ValueError("Grouped data must provide a columns attribute.")
    if "method" not in grouped.columns:
        raise ValueError("Style mapping requires a method column.")

    variant_fields = [field for field in VARIANT_FIELDS if field in grouped.columns]
    cols = ["method_label", "method"] + variant_fields
    rows = grouped.select(cols).unique().to_dicts()
    rows = sorted(rows, key=lambda row: str(row.get("method_label")))

    variants_by_method = {}
    for row in rows:
        method = row.get("method")
        key = _variant_key(row, variant_fields)
        variants_by_method.setdefault(method, set()).add(key)

    linestyle_map = {}
    for method, variants in variants_by_method.items():
        ordered = sorted(variants, key=lambda item: str(item))
        for idx, key in enumerate(ordered):
            linestyle_map[(method, key)] = LINESTYLES[idx % len(LINESTYLES)]

    style_map = {}
    for row in rows:
        method_label = row["method_label"]
        method = row.get("method")
        color = color_map[method]
        alpha = DEFAULT_ALPHA
        linewidth = DEFAULT_LINEWIDTH
        if method in BLOCK_PERM_METHODS:
            linewidth = BLOCK_PERM_LINEWIDTH.get((row.get("kappa"), row.get("s")), linewidth)
        if method == g.METHOD_FLASH_BLOCK_ROW:
            linewidth = max(linewidth, KAPPA_LINEWIDTH)
        variant_key = _variant_key(row, variant_fields)
        linestyle = linestyle_map.get((method, variant_key), LINESTYLES[0])
        style_map[method_label] = {
            "color": color,
            "alpha": alpha,
            "linewidth": linewidth,
            "linestyle": linestyle,
        }
    return style_map
