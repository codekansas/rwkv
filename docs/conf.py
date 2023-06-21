"""Sphinx configuration file."""

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "rwkv"
copyright = "2023, Benjamin Bolte"
author = "Benjamin Bolte"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",
    "sphinxcontrib.jquery",
    "myst_parser",
]

autodoc_member_order = "bysource"
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]
# autodoc_typehints = "description"
autodoc_typehints = "signature"
autoclass_content = "both"

autosummary_generate = True
autosummary_imported_members = False

github_url = "https://github.com/codekansas/rwkv"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
