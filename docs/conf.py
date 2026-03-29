# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
import os

# Add project root and src/ to sys.path
ROOT = os.path.abspath("..")  # primacore/
SRC = os.path.join(ROOT, "primacore")  # primacore/primacore/
sys.path.insert(0, SRC)

def get_version_from_json():
    import json

    version_file = os.path.join(ROOT, ".release-please-manifest.json")
    with open(version_file) as f:
        data = json.load(f)
    return data.get(".", "0.0.0")

# -- Project information -------------------------------------------------------
project = "primacore"
copyright = "2024, Jordi Sassoon"
author = "Jordi Sassoon"
release = get_version_from_json()
version = ".".join(release.split(".")[:2])

# -- General configuration -------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Generate the plots for the examples
plot_include_source = True

# autosummary
autosummary_generate = True

# autodoc
autodoc_typehints = "description"
autodoc_typehints_format = "short"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "primacore documentation"

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
}

# -- Source file suffix -------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Intersphinx mapping -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
