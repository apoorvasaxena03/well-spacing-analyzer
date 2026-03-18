"""
dashboard/components/gun_barrel.py
Builds the enhanced gun barrel go.Figure from GB + IK DataFrames.

Three trace layers:
  1. Well points  — markers+text, colored by variable (bench, operator, year, etc.)
  2. Spacing zigzag lines — right-triangle connectors with H/V/3D labels
     placed on their respective line segments (not at center)
  3. Formation top horizons (optional) — dashed lines per formation

Toggle support: show_lines, show_labels control zigzag visibility.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

_PALETTE = px.colors.qualitative.Alphabet


def empty_figure(message: str = "No data") -> go.Figure:
    """Return a blank figure with a centred annotation message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def _color_map_for(GB: pd.DataFrame, color_by: str) -> dict[str, str]:
    """Build a value → hex color mapping for a given column."""
    if color_by not in GB.columns:
        return {}
    vals = sorted(GB[color_by].dropna().astype(str).unique())
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(vals)}


def _add_well_points(
    fig: go.Figure, GB: pd.DataFrame, x_col: str, color_by: str = "bench",
) -> None:
    """Layer 1 — scatter points colored by variable, labeled with UWI."""
    col = color_by if color_by in GB.columns else "bench"
    if col not in GB.columns:
        GB = GB.copy()
        GB[col] = "Unknown"

    cmap = _color_map_for(GB, col)

    for val, grp in GB.groupby(col, dropna=False):
        val_str = str(val) if pd.notna(val) else "Unknown"
        color = cmap.get(val_str, "#7f7f7f")
        fig.add_trace(go.Scatter(
            x=grp[x_col],
            y=grp["elevation_i"],
            mode="markers+text",
            name=val_str,
            text=grp["well_i"].astype(str),
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=12, color=color, line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "TVD: %{y:,.0f} ft<br>"
                f"{'Section' if x_col == 'sectionDist' else 'Cumulative'} dist: %{{x:,.0f}} ft"
                "<extra></extra>"
            ),
        ))


def _add_spacing_zigzag(
    fig: go.Figure,
    GB: pd.DataFrame,
    IK: pd.DataFrame,
    x_col: str,
    show_lines: bool = True,
    show_labels: bool = True,
) -> None:
    """
    Layer 2 — right-triangle zigzag connectors between adjacent well pairs.

    Labels are placed ON their respective line segments:
      - H label → on horizontal dotted line (midpoint)
      - V label → on vertical dotted line (midpoint)
      - 3D label → on hypotenuse dashed line (midpoint)
    """
    if not show_lines and not show_labels:
        return

    dist3d_col = "3D_dist" if "3D_dist" in IK.columns else "dist3d"

    for idx in range(len(GB) - 1):
        wi = GB.iloc[idx]
        wk = GB.iloc[idx + 1]

        x_wi, y_wi = wi[x_col], wi["elevation_i"]
        x_wk, y_wk = wk[x_col], wk["elevation_i"]
        corner_x, corner_y = x_wk, y_wi

        if show_lines:
            # Horizontal segment
            fig.add_trace(go.Scatter(
                x=[x_wi, corner_x], y=[y_wi, corner_y],
                mode="lines",
                line=dict(color="gray", dash="dot", width=1),
                showlegend=False, hoverinfo="skip",
            ))
            # Vertical segment
            fig.add_trace(go.Scatter(
                x=[corner_x, x_wk], y=[corner_y, y_wk],
                mode="lines",
                line=dict(color="gray", dash="dot", width=1),
                showlegend=False, hoverinfo="skip",
            ))
            # Hypotenuse
            fig.add_trace(go.Scatter(
                x=[x_wi, x_wk], y=[y_wi, y_wk],
                mode="lines",
                line=dict(color="lightgray", dash="dash", width=1),
                showlegend=False, hoverinfo="skip",
            ))

        # Get spacing values from IK
        if not show_labels:
            continue

        pair = IK[
            (IK["well_i"] == wi["well_i"]) & (IK["well_k"] == wk["well_i"])
        ]
        if pair.empty:
            pair = IK[
                (IK["well_i"] == wk["well_i"]) & (IK["well_k"] == wi["well_i"])
            ]
        if pair.empty:
            continue

        h_dist = pair["horizontal_dist"].iloc[0]
        v_dist = pair["vertical_dist"].iloc[0]
        d3d = pair[dist3d_col].iloc[0]

        # H label → on horizontal line (midpoint of horizontal segment)
        fig.add_annotation(
            x=(x_wi + corner_x) / 2,
            y=y_wi,
            text=f"H: {h_dist:,.0f} ft",
            showarrow=False,
            font=dict(size=7, color="#1f77b4"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=1,
            yshift=10,
        )
        # V label → on vertical line (midpoint of vertical segment)
        fig.add_annotation(
            x=corner_x,
            y=(corner_y + y_wk) / 2,
            text=f"V: {v_dist:,.0f} ft",
            showarrow=False,
            font=dict(size=7, color="#d62728"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=1,
            xshift=35,
        )
        # 3D label → on hypotenuse (midpoint of diagonal)
        fig.add_annotation(
            x=(x_wi + x_wk) / 2,
            y=(y_wi + y_wk) / 2,
            text=f"3D: {d3d:,.0f} ft",
            showarrow=False,
            font=dict(size=7, color="#2ca02c"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=1,
            yshift=-10,
        )


def _add_formation_tops(
    fig: go.Figure,
    df_formation_tops: pd.DataFrame,
    GB: pd.DataFrame,
    x_col: str,
) -> None:
    """Layer 3 (optional) — dashed formation horizon lines."""
    x_min = GB[x_col].min()
    x_max = GB[x_col].max()

    for formation, grp in df_formation_tops.groupby("formation"):
        merged = grp.merge(
            GB[["well_i", x_col]],
            left_on="uwi", right_on="well_i",
            how="inner",
        ).sort_values(x_col)

        if merged.empty:
            mean_tvd = grp["top_tvd"].mean()
            xs = [x_min, x_max]
            ys = [mean_tvd, mean_tvd]
        else:
            xs = [x_min] + list(merged[x_col]) + [x_max]
            ys = [merged["top_tvd"].iloc[0]] + list(merged["top_tvd"]) + [merged["top_tvd"].iloc[-1]]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=str(formation),
            line=dict(width=1.5, dash="longdash"),
            hovertemplate=f"{formation}: %{{y:,.0f}} ft TVD<extra></extra>",
        ))


def build_gun_barrel_figure(
    GB: pd.DataFrame,
    IK: pd.DataFrame,
    x_col: str = "sectionDist",
    color_by: str = "bench",
    show_lines: bool = True,
    show_labels: bool = True,
    df_formation_tops: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Build the enhanced gun barrel figure.

    Args:
        GB: Output of compute_gun_barrel().
        IK: Filtered IK spacing pairs.
        x_col: 'sectionDist' (centered) or 'cum_dist' (from reference).
        color_by: Column to color wells by (bench, operator, spud_year, etc.)
        show_lines: Show zigzag connector lines.
        show_labels: Show H/V/3D distance labels.
        df_formation_tops: Optional formation top data.
    """
    if GB.empty:
        return empty_figure("No gun barrel data available.")

    fig = go.Figure()

    _add_well_points(fig, GB, x_col, color_by=color_by)
    _add_spacing_zigzag(fig, GB, IK, x_col, show_lines=show_lines, show_labels=show_labels)

    if df_formation_tops is not None and not df_formation_tops.empty:
        _add_formation_tops(fig, df_formation_tops, GB, x_col)

    x_label = "Section Distance (ft)" if x_col == "sectionDist" else "Cumulative Distance (ft)"

    fig.update_layout(
        title="Gun Barrel — Cross-Section View",
        xaxis=dict(
            title=x_label,
            zeroline=False,
            tickformat=",",
        ),
        yaxis=dict(
            title="Depth TVD (ft)",
            autorange="reversed",
            tickformat=",",
        ),
        legend=dict(title="Bench / Formation", orientation="h", y=-0.15),
        hovermode="closest",
        template="plotly_white",
        margin=dict(t=50, b=80, l=60, r=20),
    )
    return fig
