"""Sphinx configuration for the OTP-FM documentation site.

Built locally via ``pixi run docs-html`` (or ``sphinx-build -b html docs
docs/_build/html``) and on Read the Docs via ``.readthedocs.yaml``.
"""

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOCS_DIR.parent

# Mirror the top-level notebooks into the docs tree so myst-nb can render them
# under ``docs/tutorials/``. The copies are gitignored; the originals in
# ``notebooks/`` remain the source of truth.
_NB_SRC = _REPO_ROOT / "notebooks"
_NB_DST = _DOCS_DIR / "tutorials"
if _NB_SRC.is_dir():
    _NB_DST.mkdir(exist_ok=True)
    for _nb in sorted(_NB_SRC.glob("*.ipynb")):
        _target = _NB_DST / _nb.name
        if not _target.exists() or _target.stat().st_mtime < _nb.stat().st_mtime:
            shutil.copy2(_nb, _target)

project = "OTP-FM"
author = "Raghav Kansal"
copyright = "2026, Bexorg, Inc"  # noqa: A001

try:
    release = importlib.metadata.version("otpfm")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # myst_nb pulls in myst_parser internally; listing both triggers a
    # double-setup that crashes on Sphinx's footnote-detector transform.
    "myst_nb",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]
# alpha listing pairs naturally with author-year inline cites:
# bibliography shows ``[Kan26]`` prefixes that match what readers see inline.
bibtex_default_style = "alpha"
# Renders :cite:t: as "Kansal et al. (2026)" and :cite:p: as "(Kansal et al., 2026)".
# sphinxcontrib-bibtex ships only square-bracket variants of author_year out of
# the box; we register a round-paren one below.
bibtex_reference_style = "author_year_round"


def _register_author_year_round_style() -> None:
    """Register an ``author_year_round`` reference style with round brackets."""
    from dataclasses import dataclass, field

    import sphinxcontrib.bibtex.plugin
    from sphinxcontrib.bibtex.style.referencing import BracketStyle
    from sphinxcontrib.bibtex.style.referencing.author_year import (
        AuthorYearReferenceStyle,
    )

    def _round() -> BracketStyle:
        return BracketStyle(left="(", right=")")

    @dataclass
    class AuthorYearRoundReferenceStyle(AuthorYearReferenceStyle):
        bracket_parenthetical: BracketStyle = field(default_factory=_round)
        bracket_textual: BracketStyle = field(default_factory=_round)
        bracket_author: BracketStyle = field(default_factory=_round)
        bracket_label: BracketStyle = field(default_factory=_round)
        bracket_year: BracketStyle = field(default_factory=_round)

    sphinxcontrib.bibtex.plugin.register_plugin(
        "sphinxcontrib.bibtex.style.referencing",
        "author_year_round",
        AuthorYearRoundReferenceStyle,
    )


_register_author_year_round_style()

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
    # Contributor-facing README for the docs directory; not part of the site.
    "README.md",
]

# myst-nb registers the parsers for ``.md`` and ``.ipynb`` itself; we just need
# to keep the default ``.rst`` source suffix.
source_suffix = {
    ".rst": "restructuredtext",
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"OTP-FM {version}"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autoclass_content = "class"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Tutorial notebooks ship with cleared outputs; rendering them with execution
# would require the full training stack on the RTD build image. Re-execute and
# commit notebooks locally if you want rich plots in the rendered docs.
nb_execution_mode = "off"
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Quiet a few common warnings without hiding real issues:
# - ``myst.header`` fires on the level-skipping headings used in our READMEs.
# - ``myst.xref_missing`` fires on placeholder ``[text]()`` citation links in
#   the upstream README/REPRODUCIBILITY pages.
suppress_warnings = ["myst.header", "myst.xref_missing"]
