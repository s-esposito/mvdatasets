import os
import sys
sys.path.insert(0, os.path.abspath('../mvdatasets'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MVDatasets'
copyright = '2025, Stefano Esposito, Andreas Geiger'
author = 'Stefano Esposito, Andreas Geiger'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Enables automatic docstring extraction
    'sphinx.ext.napoleon',  # Supports Google-style and NumPy-style docstrings (optional)
    'sphinxcontrib.mermaid',  # Enables Mermaid diagrams
    'sphinxcontrib.bibtex',  # Enables BibTeX citations
    'myst_parser',  # Enables Markdown parsing
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/logo.png'

# bibtex
bibtex_bibfiles = ['refs.bib']

# markdown
source_suffix = ['.rst', '.md']