import os
import sys

sys.path.insert(0, os.path.abspath("../mvdatasets"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MVDatasets"
copyright = "2025, Stefano Esposito, Andreas Geiger"
author = "Stefano Esposito, Andreas Geiger"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Enables automatic docstring extraction
    "sphinx.ext.napoleon",  # Supports Google-style and NumPy-style docstrings
    "sphinxcontrib.mermaid",  # Enables Mermaid diagrams
    "sphinxcontrib.bibtex",  # Enables BibTeX citations
    # "sphinxcontrib.osexample",  # Enable tabs for multiple code examples
    "sphinx.ext.autosectionlabel",  # Enables autosectionlabel
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'analytics_id': 'G-1T0QG65M3V',
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'black',
    'flyout_display': 'hidden',
    'version_selector': False,
    'language_selector': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "../imgs/MVD.png"

# bibtex
bibtex_bibfiles = ["refs.bib"]
