"""Sphinx configuration."""

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Project Lighthouse Anonymize'
author = 'Airbnb Anti-Discrimination & Equity Engineering Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_copyright = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_typehints = 'signature'
typehints_document_rtype = False
always_document_param_types = False
typehints_use_signature = True
typehints_use_signature_return = False
