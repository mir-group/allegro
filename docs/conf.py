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
    "sphinx_rtd_theme",
    "myst_parser",
]
myst_enable_extensions = [
    "html_admonition",
    "dollarmath",  # "amsmath", # to parse Latex-style math
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autosummary_generate = True
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_logo = "../logo.png"
html_theme_options = {
    "logo_only": True,
}


def setup(app):
    app.add_css_file("custom.css")
