"""Ligand-receptor pair collections used by CCI pair extraction."""


def _normalize_collection_name(name: str) -> str:
    return name.lower().replace(" ", "_")


# Human-readable collection name -> LR pair list
COLLECTIONS: dict[str, list[tuple[str, str]]] = {
    "LIANA Embryo Signaling": [
        ("TIMP1", "CD63"),
        ("MDK", "SDC2"),
        ("PTN", "NCL"),
        ("HMGB1", "CXCR4"),
        ("LUM", "ITGB1"),
        ("APOE", "LSR"),
        ("HAPLN1", "PRTG"),
        ("GPC3", "LRP1"),
        ("APP", "TSPAN15"),
        ("SFRP1", "FZD2"),
    ],
    "LIANA Cancer Signaling": [
        ("APP", "CD74"),
        ("CALM1", "AQP1"),
        ("VIM", "CD44"),
        ("B2M", "KLRD1"),
        ("GNAI2", "CAV1"),
        ("HMGB1", "THBD"),
        ("MRC1", "PTPRC"),
        ("VWF", "ITGB1"),
        ("LGALS3", "ENG"),
        ("COL4A1", "CD47"),
    ],
    "Channels 1 2 3": [
        ("CCL9", "CCR1"),
        ("CD274", "PDCD1"),
        ("IL23A", "IL23R"),
    ],
    "Channels 1 2": [
        ("CCL9", "CCR1"),
        ("CD274", "PDCD1"),
    ],
    "Channels 1 3": [
        ("CCL9", "CCR1"),
        ("IL23A", "IL23R"),
    ],
    "Channels 2 3": [
        ("CD274", "PDCD1"),
        ("IL23A", "IL23R"),
    ],
    "LIANA Light Mouse Signaling": [
        ("APP", "APLP1"),
        ("PSAP", "GPR37L1"),
        ("GNAS", "ADCY1"),
        ("PTN", "PTPRB"),
        ("CD47", "SIRPA"),
        ("RTN4", "LINGO1"),
        ("BCAN", "NRCAM"),
        ("CXCL12", "ITGB1"),
        ("COL4A1", "CD47"),
        ("NXPH1", "NRXN1"),
    ],
    "LIANA Immune Signaling": [
        ("CCL5", "SDC4"),
        ("CD14", "ITGB2"),
        ("APP", "CD74"),
        ("INHBA", "ACTR2"),
        ("CALM1", "FAS"),
        ("MMP12", "PLAUR"),
        ("ANXA1", "FPR1"),
        ("TNF", "TNFRSF1B"),
        ("LGALS1", "CD69"),
        ("ANXA2", "TLR2"),
    ],
}

# Normalized key -> canonical collection name
NORMALIZED_NAME_TO_CANONICAL: dict[str, str] = {
    "liana_embryo_signaling": "LIANA Embryo Signaling",
    "liana_cancer_signaling": "LIANA Cancer Signaling",
    "channels_1_2_3": "Channels 1 2 3",
    "channels_1_2": "Channels 1 2",
    "channels_1_3": "Channels 1 3",
    "channels_2_3": "Channels 2 3",
    "liana_light_mouse_signaling": "LIANA Light Mouse Signaling",
    "liana_immune_signaling": "LIANA Immune Signaling",
}


def get_lr_pairs(collection_name: str) -> list[tuple[str, str]]:
    """Get ligand-receptor pairs by collection name (case-insensitive)."""
    key = _normalize_collection_name(collection_name)
    canonical = NORMALIZED_NAME_TO_CANONICAL.get(key)
    if canonical is None:
        available = sorted(NORMALIZED_NAME_TO_CANONICAL.keys())
        raise KeyError(
            f"Collection {collection_name!r} not found. Available: {available}"
        )
    return COLLECTIONS[canonical]
