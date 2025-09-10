# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PETGEM'
copyright = '2025, Castillo Reyes, Octavio'
author = 'Castillo Reyes, Octavio'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'breathe',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Breathe configuration
breathe_projects = {
    "PETGEM": "../doxygen/xml/"
}
breathe_default_project = "PETGEM"
breathe_domain_by_extension = {
    "h": "c",
    "c": "c",
}
breathe_implementation_filename_extensions = ['.c', '.cc', '.cpp']