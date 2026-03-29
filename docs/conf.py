# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add the parent directory to the path so we can import primacore
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -------------------------------------------------------
project = "primacore"
copyright = "2024, Jordi Sassoon"
author = "Jordi Sassoon"
release = "0.1.4"

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
    ".rst": None,
    ".md": "myst-nb",
}

# -- Intersphinx mapping -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
