# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MethylBERT'
copyright = '2025, Yunhee Jeong'
author = 'Yunhee Jeong'
release = '2.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.mathjax'] # for Markdown and Latex

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Python read-the-docs template is used
# https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "relbarbgcolor": "#95a5a6"
}