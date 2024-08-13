from visdom import Visdom

vis = Visdom()

svgstr ="""
<svg height="300" width="300">
<ellipse cx="80" cy="80" rx="50" ry="30"
style="fill:red;stroke:purple;stroke-width:2;" />
Sorry, your browser does not support inline SVG. </svg>
"""

vis.svg(
    svgstr = svgstr,
    opts=dict(title = "svg picture")
)