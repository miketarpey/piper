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
import inspect
from pathlib import Path

current_dir = Path(".").absolute()
module_dir = current_dir.parents[1]
sys.path.insert(0, module_dir.as_posix())

# -- Project information -----------------------------------------------------

project = 'Piper'
copyright = '2021, Mike Tarpey'
author = 'Mike Tarpey'

# The full version, including alpha/beta/rc tags
from piper.version import __version__
release = __version__

# -- General configuration ---------------------------------------------------



# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_markdown_tables",
    "sphinx.ext.linkcode",
    "numpydoc",  # handle NumPy documentation formatted docstrings
]


source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst', '.md']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


autosummary_generate = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_logo = "../source/_static/piper_logo.png"

html_theme_options = {
    "search_bar_position": "navbar",
    'display_version': True,
}

html_theme_options = {

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/miketarpey/piper",
            "icon": "fab fa-github-square",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
        fn = Path(fn).name

    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    return f"https://github.com/miketarpey/piper/tree/master/piper/{fn}{linespec}"
