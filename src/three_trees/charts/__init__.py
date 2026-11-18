"""Plotly charts for tournament pathways.

The "simple_white" template draws shapes with line width 0 and opacity 0.3 by default, which leaves reference
lines added without a width (add_vline, add_hline) invisible; every chart here uses that template, so its shape
defaults are set to a visible line once, on import.
"""

import plotly.io as pio

pio.templates["simple_white"].layout.shapedefaults = {"line": {"width": 1.5}, "opacity": 1}
