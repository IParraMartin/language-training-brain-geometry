"""Sphinx configuration for the paper page and documentation.

The site is a paper-style home page (``index.md``) followed by the reproduction, corpora, benchmark and
code-guide pages, rendered with the local ``minimal`` theme in ``_themes/``. Only the documentation
tools are needed to build it: ``uv run --only-group docs sphinx-build -b html -W docs site``.
"""

from __future__ import annotations

project = "Language Training Shapes Brain-Aligned Geometry"
author = "Iñigo Parra"
copyright = "2026, Iñigo Parra"
release = "1.0.0"
repository_url = "https://github.com/IParraMartin/language-training-brain-geometry"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton"
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "attrs_block",
    "attrs_inline"
]
myst_heading_anchors = 3

# local theme in the style of academic project pages: no sidebars and a centered column
html_theme = "minimal"
html_theme_path = ["_themes"]
html_theme_options = {"repository_url": repository_url}
html_title = project
html_short_title = "Brain-Aligned Geometry"
html_static_path = ["_static"]
html_show_sourcelink = False
html_copy_source = False
html_baseurl = "https://iparramartin.github.io/language-training-brain-geometry/"
html_permalinks_icon = "#"

copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
