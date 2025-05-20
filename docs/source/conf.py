# -- Project information -----------------------------------------------------
import os
import sys
from datetime import date

# Add SpatioloJI to the Python path
sys.path.insert(0, os.path.abspath('../..'))

# Try to import the package version
try:
    from SpatioloJI import __version__ as version
except ImportError:
    version = '0.1.0'  # Default version if import fails

project = 'SpatioloJI'
copyright = f'{date.today().year}, SpatioloJI Developers'
author = 'SpatioloJI Developers'
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate API documentation
    'sphinx.ext.autosummary',    # Generate summary tables
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',    # Link to other projects' documentation
    'sphinx.ext.mathjax',        # Render math via MathJax
    'sphinx.ext.githubpages',    # Create .nojekyll file for GitHub Pages
    'nbsphinx',                  # Jupyter notebook integration
    'sphinx_copybutton',         # Add "copy" button to code blocks
    'sphinx_autodoc_typehints',  # Use Python type hints in documentation
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude from source
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme
html_static_path = ['_static']
html_logo = '_static/logo.svg'   # Assuming you copy your logo to _static

# Favicon
html_favicon = '_static/favicon.ico'  # You'll need to create this

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = 'auto'        # Execute notebooks during build
nbsphinx_allow_errors = True    # Don't build if notebooks have errors
nbsphinx_execute = 'never'

# -- Extension configurations -------------------------------------------------
autodoc_member_order = 'bysource'  # Order members as they appear in the source
autosummary_generate = True      # Generate API docs automatically

# Add mappings for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
}

# -- HTML theme settings -----------------------------------------------------
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'style_external_links': True,
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Custom code to run during documentation building ------------------------
def setup(app):
    # Add custom CSS file
    app.add_css_file('custom.css')
