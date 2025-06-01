# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Allegro"
copyright = "2025, MIR"
author = "MIR"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]
myst_enable_extensions = [
    "html_admonition",
    "dollarmath",  # "amsmath", # to parse Latex-style math
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autosummary_generate = True
source_suffix = [".rst", ".md"]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "e3nn": ("https://docs.e3nn.org/en/stable/", None),
    "torchmetrics": ("https://lightning.ai/docs/torchmetrics/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_favicon = "favicon.png"
html_logo = "../logo.png"
html_theme_options = {
    "sidebar_hide_name": True,
}


def process_docstring(app, what, name, obj, options, lines):
    """For pretty printing sets and dictionaries of data fields."""
    if isinstance(obj, set) or isinstance(obj, dict):
        lines.clear()  # Clear existing lines to prevent repetition


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
