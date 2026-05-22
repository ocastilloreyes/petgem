# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup for autodoc --------------------------------------------------
# The Python helpers live in the `utils` package and are imported under the
# distribution name `petgem` (see tests/conftest.py). Make both importable for
# autodoc, and stand in lightweight mocks for the heavy runtime dependencies
# that are not installed in the docs build environment (Read the Docs has no
# PETSc/HDF5). Mocks are used ONLY when the real module is unavailable, so a
# local build with the full stack keeps the real packages.
import importlib
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../..'))   # repo root (docs/source/../..)

for _mod in ('numpy', 'h5py', 'meshio', 'petsc4py', 'petsc4py.PETSc'):
    try:
        importlib.import_module(_mod)
    except Exception:
        sys.modules[_mod] = MagicMock()

# Expose `utils` as `petgem` so autodoc can resolve `petgem.*`.
import utils as _petgem_pkg          # noqa: E402
sys.modules.setdefault('petgem', _petgem_pkg)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PETGEM'
copyright = '2025, Castillo-Reyes, Octavio'
author = 'Castillo-Reyes, Octavio'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.mathjax',
	'breathe',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/petgem_logo.png"

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