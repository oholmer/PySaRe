# Configuration file for the Sphinx documentation builder.
import sys
import os

# -- Project information

project = "PySaRe"
copyright = "2024, Olov Holmer"
author = "Olov Holmer"

release = "0.1.2"
version = "0.1.2"

# -- General configuration
pygments_style = "sphinx"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.napoleon"
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../scr"))
sys.path.insert(0, os.path.abspath("../src/pysare"))
