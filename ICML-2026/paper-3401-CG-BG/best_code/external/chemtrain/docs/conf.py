# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath('..'))

from datetime import datetime
from chemtrain.version import __version__ as chemtrain_version

# -- Project information -----------------------------------------------------

project = 'Chemtrain'
copyright = (f'{datetime.now().year}, Multiscale Modeling of Fluid Materials, '
             f'TU Munich')
author = 'Multiscale Modeling of Fluid Materials'
#
release = chemtrain_version


# -- Build docs with Doxygen -------------------------------------------------

subprocess.call("doxygen Doxyfile", shell=True)

breathe_projects = { "chemtrain-deploy": "doxygen/xml" }
breathe_default_project = "chemtrain-deploy"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_remove_toctrees',
    'sphinxcontrib.mermaid',
    'myst_nb',
    'breathe',
]

napoleon_numpy_docstring = False

napoleon_attr_annotations = True

napoleon_use_ivar= True

autosummary_generate = True

# Do not write chemtrain before every subpackage/function/etc.
add_module_names = False

# Remove unnecessarily deep tocs for trainers

remove_from_toctrees = [
    "api/_autosummary/_autosummary/*",
    "api/**/_autosummary/*",
    "jax_md_mod/**/_autosummary/*"
]

templates_path = ["_templates"]

autodoc_mock_imports = [
    'e3nn_jax', 'mace_jax'
]


source_suffix = ['.rst', '.ipynb']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'jax': ('https://jax.readthedocs.io/en/latest', None),
    'jax-sgmc': ('https://jax-sgmc.readthedocs.io/en/latest', None),
    'jax-md': ('https://jax-md.readthedocs.io/en/main', None)
}

# Jupyter options
nb_execution_mode = "auto"
nb_execution_timeout = -1
nb_execution_excludepatterns = [
  # Require long computations
  'examples/*',
]

os.environ["DATA_PATH"] = "../../examples/data"

myst_footnote_transition = False

# -- MathJax ------------------------------------------------------------------

mathjax3_config = {
  "tex": {
    "inlineMath": [['$', '$'], ['\\(', '\\)']]
  },
  "svg": {
    "fontCache": 'global'
  }
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": "https://github.com/tummfm/chemtrain",
    "use_repository_button": True,
    "home_page_in_toc": True
}

html_title = "chemtrain"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# Add global definitions of custom roles

with open("_templates/global.rst", "r") as f:
    rst_prolog = f.read()
