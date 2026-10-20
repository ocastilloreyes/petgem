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
html_logo = "_static/brand/01_AF_PETGEM_Logo_No_Claim_Gradient.svg"
html_theme_options = {
    "logo_only": True,
    "style_nav_header_background": "#180E35",
}
html_css_files = ["brand.css"]

# -- Options for LaTeX/PDF output --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

latex_engine = 'xelatex'
latex_use_xindy = False
latex_documents = [
    ('index', 'petgem.tex', 'PETGEM', author, 'manual'),
]
latex_additional_files = [
    '_latex/01_AF_PETGEM_Logo_Asset_Gradient.pdf',
    '_latex/03_PETGEM_Lines_Gradient.pdf',
]

_latex_preamble = r'''
\usepackage{tikz}
\definecolor{petgemgreen}{HTML}{1DB681}
\definecolor{petgemblue}{HTML}{3E5AA9}
\definecolor{petgembg}{HTML}{180E35}

% Part pages
\titleformat{\part}[display]
  {\centering\sffamily\bfseries\Huge\color{TitleColor}}
  {\partname~\thepart}{20pt}{}

% Gradient rule
\newcommand{\petgemrule}[2]{%
  \tikz\shade[left color=petgemgreen, right color=petgemblue]
    (0,0) rectangle (#1,#2);}

% Header and footer rules
\makeatletter
\renewcommand{\headrule}{%
  \ifdim\headrulewidth>\z@
    \vbox to\z@{\vss\hbox{\petgemrule{\headwidth}{1pt}}}\vskip-\headrulewidth
  \fi}
\renewcommand{\footrule}{%
  \ifdim\footrulewidth>\z@
    \vbox to\z@{\vss\hbox{\petgemrule{\headwidth}{1pt}}}\vskip\footruleskip
  \fi}
\makeatother
'''

_latex_maketitle = r'''
\makeatletter
\begin{titlepage}
\begin{tikzpicture}[remember picture, overlay]
  \fill[petgembg] (current page.south west) rectangle (current page.north east);
  \node[anchor=south, inner sep=0pt] at (current page.south)
    {\includegraphics[width=\paperwidth]{03_PETGEM_Lines_Gradient.pdf}};
  \node[anchor=center, inner sep=0pt] (logo) at ([yshift=7cm]current page.center)
    {\includegraphics[width=0.8\paperwidth]{01_AF_PETGEM_Logo_Asset_Gradient.pdf}};
  \node[anchor=north, inner sep=0pt] (bar) at ([yshift=-0.5cm]logo.south)
    {\petgemrule{0.71\paperwidth}{3pt}};
  \node[anchor=north east, inner sep=0pt, align=right, font=\sffamily]
    at ([yshift=-0.8cm]bar.south east)
    {{\color{petgemgreen}\bfseries\Huge USER MANUAL}\\[1.2cm]
     {\color{white}\Large \py@release}\\[0.4cm]
     {\color{white}\large \@author}\\[0.4cm]
     {\color{white}\large \@date}};
\end{tikzpicture}
\end{titlepage}
\makeatother
'''

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'extraclassoptions': 'openany,oneside',
    'geometry': r'\usepackage[headheight=20pt]{geometry}',
    'preamble': _latex_preamble,
    'maketitle': _latex_maketitle,
    'tableofcontents': r'\renewcommand{\contentsname}{Contents}\sphinxtableofcontents',
    'sphinxsetup': (
        'TitleColor={HTML}{3E5AA9}, '
        'InnerLinkColor={HTML}{3E5AA9}, '
        'OuterLinkColor={HTML}{3E5AA9}'
    ),
}

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