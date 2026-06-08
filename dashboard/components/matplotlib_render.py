"""
dashboard/components/matplotlib_render.py
Utility for rendering matplotlib figures as Dash html.Img components.
"""

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
from dash import html

# Use non-interactive backend so figures never pop up in the server process
matplotlib.use("Agg")


def fig_to_base64_img(fig, *, dpi: int = 150, close: bool = True) -> html.Img:
    """Convert a matplotlib Figure to an html.Img with inline base64 PNG src.

    Args:
        fig: matplotlib Figure object.
        dpi: Resolution for the PNG render.
        close: Whether to close the figure after rendering (prevents memory leak).

    Returns:
        dash.html.Img with src="data:image/png;base64,..." and width="100%".
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={"width": "100%", "maxWidth": "900px"},
    )


def capture_all_open_figs(*, dpi: int = 150) -> list[html.Img]:
    """Capture all currently open matplotlib figures as html.Img components.

    Useful when a library function creates multiple figures via plt.figure()
    and you want to grab them all (e.g. debug_pair_spacing).

    Returns:
        List of html.Img components, one per open figure.
    """
    images = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        images.append(fig_to_base64_img(fig, dpi=dpi, close=True))
    return images
