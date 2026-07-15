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
import datetime

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# import pydata_sphinx_theme

# -- Project information -----------------------------------------------------

project = "APPFL"
copyright = (
    "2022-%d, UChicago Argonne, LLC and the APPFL Development Team"
    % datetime.date.today().year
)
author = "The APPFL Development Team"

git_describe_version = (
    subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "nbsphinx",
    "sphinx_contributors",
    "sphinx_copybutton",
    "sphinx_design",
]

autodoc_mock_imports = [
    "torch",
    "omegaconf",
    "grpc",
    "numpy",
    "mpi4py",
    "zfpy",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_data"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_favicon = "_static/favicon.ico"
html_logo = "_static/logo_yellow_with_name.png"
html_last_updated_fmt = r"%Y-%m-%dT%H:%M:%S%z (" + git_describe_version + ")"
pygments_style = "friendly"
pygments_dark_style = "monokai"  # this is a furo-specific option
html_theme_options = {
    # "content_footer_items": ["last-updated"],
    "footer_items": ["copyright", "sphinx-version", "last-updated"],
    "show_toc_level": 2,
    # "use_edit_page_button": True,
    # "icon_links": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/APPFL/APPFL",
    #         "icon": "fab fa-github",
    #     },
    #     {
    #         "name": "Discord",
    #         "url": "https://discord.gg/bBW56EYGUS",
    #         "icon": "fab fa-discord",
    #     },
    #     {
    #         "name": "YouTube",
    #         "url": "https://www.youtube.com/channel/UCzwiJboiJW3dLI0UndnDy5g",
    #         "icon": "fab fa-youtube",
    #     },
    # ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "github-link.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# html_copy_source = False

nbsphinx_execute = "never"
