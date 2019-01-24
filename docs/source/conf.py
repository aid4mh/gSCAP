# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from recommonmark.parser import CommonMarkParser


# -- Project information -----------------------------------------------------
project = 'gSCAP'
copyright = '2018, Luke Waninger, Abhishek Pratap'
author = 'Luke Waninger, Abhishek Pratap'
version = 'latest'
release = '0'


# -- General configuration ---------------------------------------------------
# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = ['.rst', '.md']
source_parsers = {'.md': CommonMarkParser}
master_doc = 'index'
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_sidebars = {
    '**': [
        'globaltoc.html'
        'localtoc.html',
        'searchbox.html',
    ]
}

htmlhelp_basename = 'brightenv2doc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'brightenv2.tex', 'brightenv2 Documentation',
     'Abhishek Pratap, Luke Waninger', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'brightenv2', 'brightenv2 Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'brightenv2', 'brightenv2 Documentation',
     author, 'brightenv2', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------
todo_include_todos = True
