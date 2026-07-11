#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Learning Parities with Product Networks'
copyright = '2026, Orange S.A.'
author = 'Guillaume Larue, Louis-Adrien Dufrène, Quentin Lampin, Hadi Ghauch, Ghaya Rekaya'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#ff7900",        # Orange primary
        "color-brand-content": "#ff7900",        # Orange for links
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f8f9fa",
    
    },
    "dark_css_variables": {
        "color-brand-primary": "#ffa366",        # Lighter orange for dark
        "color-brand-content": "#ffa366",
        "color-background-primary": "#1e1e1e",
        "color-background-secondary": "#252525",
    },
    "source_repository": "https://github.com/Orange-OpenSource/learning-parities-with-product-networks",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Orange-OpenSource/learning-parities-with-product-networks",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
        {
            "name": "arXiv",
            "url": "https://arxiv.org/abs/2605.28612",
            "html": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" stroke="none" width="16" height="16"><path d="M3 3v18h18V3H3zm9 15H8v-2h4v2zm4-4H8v-2h8v2zm0-4H8V8h8v2z"/></svg>',
            "class": "",
        },
    ],
}

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_title = "Learning Parities with Product Networks using GD"
# Copy additional files (paper HTML and assets)
import shutil
from pathlib import Path

# -- Sync source files from project root into docs/ before build -------------
# This ensures docs/notebooks/ and docs/studies/ always reflect the latest
# versions from the project root, avoiding manual copy and desync issues.
# Only .ipynb and .py files are synced; .rst index files in docs/ are kept.

def _sync_source_files(app):
    """Copy notebooks and study scripts from the project root into docs/."""
    root = Path(app.srcdir).parent  # project root

    # notebooks/*.ipynb  →  docs/notebooks/*.ipynb
    nb_src = root / 'notebooks'
    nb_dst = Path(app.srcdir) / 'notebooks'
    if nb_src.exists():
        nb_dst.mkdir(exist_ok=True)
        for f in nb_src.glob('*.ipynb'):
            shutil.copy2(f, nb_dst / f.name)

    # studies/*.py  →  docs/studies/*.py
    st_src = root / 'studies'
    st_dst = Path(app.srcdir) / 'studies'
    if st_src.exists():
        st_dst.mkdir(exist_ok=True)
        for f in st_src.glob('*.py'):
            shutil.copy2(f, st_dst / f.name)


def _fix_notebook_study_links(app, exception):
    """Post-build: rewrite ../studies/run_study_X.py hrefs in notebook HTML pages
    to ../studies/study_X.html so they point to the documentation page instead of
    the raw script (which is not served in the HTML output).

    The notebooks keep the original ../studies/run_study_X.py link so that the
    file is navigable from an IDE / code viewer, while the built HTML gets a
    working hyperlink to the corresponding study docs page.
    """
    if exception is not None:
        return
    import re
    notebooks_dir = Path(app.outdir) / 'notebooks'
    if not notebooks_dir.exists():
        return
    # e.g. ../studies/run_study_A_bis.py  →  ../studies/study_A_bis.html
    pattern = re.compile(r'href="\.\./studies/run_(study_[A-Za-z_]+)\.py"')
    for html_file in notebooks_dir.glob('*.html'):
        text = html_file.read_text(encoding='utf-8')
        new_text = pattern.sub(r'href="../studies/\1.html"', text)
        if new_text != text:
            html_file.write_text(new_text, encoding='utf-8')


def _copy_paper_files(app, exception):
    """Copy paper assets to the build output (post-build)."""
    if exception is None:
        paper_src = Path(app.srcdir) / 'paper'
        paper_dst = Path(app.outdir) / 'paper'

        # Copy paper_content.html (cleaned body content)
        if (paper_src / 'paper_content.html').exists():
            shutil.copy2(paper_src / 'paper_content.html', paper_dst / 'paper_content.html')

        # Copy assets directory
        assets_src = paper_src / 'assets'
        assets_dst = paper_dst / 'assets'
        if assets_src.exists():
            shutil.copytree(assets_src, assets_dst, dirs_exist_ok=True)


def setup(app):
    app.connect('builder-inited', _sync_source_files)
    app.connect('build-finished', _fix_notebook_study_links)
    app.connect('build-finished', _copy_paper_files)

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",  # Enable $...$ and $$...$$ math syntax
]

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = 'never'
