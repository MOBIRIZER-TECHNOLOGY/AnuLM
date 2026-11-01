One module per part (`part_i_ii.py`, `part_v.py`, ...). Every function named `fig_*` returns the
file name of the SVG it wrote; `python book/tools/make_figures.py part_v` runs one module.
Import the helpers with `from make_figures import text, rect, line, svg, close, INK, MUTED, ...`.
