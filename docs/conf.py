#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

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
import shutil
import glob
sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)
import sphinx_rtd_theme
from datetime import datetime

# -- Project information -----------------------------------------------------

project = 'Intel® Transfer Learning Tool'
author = 'Intel Corporation'
copyright = '2022-' + str(datetime.now().year) + u', ' + author

# The full version, including alpha/beta/rc tags
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_click',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv3', 'README.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_last_updated_fmt = '%b %d, %Y'
html_show_sphinx = False
html_favicon = 'images/favicon-intel-32x32.png'

html_static_path = ['_static']
templates_path = ['_templates']

def setup(app):
   app.add_css_file("tlt-custom.css")
   app.add_js_file("tlt-custom.js")

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
autodoc_member_order = 'bysource'
nbsphinx_execute = 'never'
nbsphinx_prolog = """
:orphan:

"""
myst_heading_anchors = 2
suppress_warnings = ["myst.xref_missing", "myst.header"]

# ask the myst parser to process <img> tags so Sphinx can handle the properly
myst_enable_extensions = [ "html_image" ]
