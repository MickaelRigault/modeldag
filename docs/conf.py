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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'modeldag'
copyright = '2023, Mickael Rigault'
author = 'Mickael Rigault'

# The full version, including alpha/beta/rc tags
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
for x in os.walk('../skysurvey'):
  sys.path.insert(0, x[0])


from skysurvey import *
from skysurvey import target



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    # extra
    "numpydoc",
    'myst_nb',
    "nbsphinx",
    'sphinx_copybutton'
    ]

nbsphinx_execute = 'never'
nb_execution_mode = "off"

autoclass_content = "both"              # Insert class and __init__ docstrings
autodoc_member_order = "bysource"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    #'emcee': ('https://emcee.readthedocs.io/en/latest', None),
}

    
    
# Add any paths that contain templates here, relative to this directory.
source_suffix = ['.rst', '.ipynb', '.md']


html_logo = '_static/modeldag_logo.png'

#html_favicon = '_static/favicon.png'

#html_theme = 'alabaster'
html_theme = 'sphinx_book_theme'
#html_theme = "pydata_sphinx_theme"

html_theme_options = {
    'logo_only': True,
    'show_toc_level': 2,
    'repository_url': f'https://github.com/MickaelRigault/{project}',
    'use_repository_button': True,     # add a "link to repository" button
}


html_static_path = ['_static']
