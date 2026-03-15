"""
dashboard/components/gun_barrel.py
Builds the enhanced gun barrel go.Figure from GB + IK DataFrames.

Three trace layers:
  1. Well points  — markers+text, colored by bench
  2. Spacing zigzag lines — right-triangle connectors annotated with
     horizontal_dist, vertical_dist, dist3d
  3. Formation top horizons (optional) — dashed lines per formation
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Default bench → color map (extend as needed)
BENCH_COLORS: dict[str, str] = {
    "WOLFCAMP A":   "#1f77b4",
    "WOLFCAMP B":   "#ff7f0e",
    "WOLFCAMP C":   "#2ca02c",
    "WOLFCAMP D":   "#d62728",
    "SPRABERRY":    "#9467bd",
    "DEAN":         "#8c564b",
    "CLEARFORK":    "#e377c2",
    "BARNETT":      "#7f7f7f",
    "ATOKA":        "#bcbd22",
    "STRAWN":       "#17becf",
}


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


def _bench_color(bench: str) -> str:
    return BENCH_COLORS.get(str(bench).upper(), "#7f7f7f")


def _add_well_points(fig: go.Figure, GB: pd.DataFrame, x_col: str) -> None:
    """Layer 1 — scatter points per bench, labeled with well name + date."""
    for bench, grp in GB.groupby("bench", dropna=False):
        color = _bench_color(str(bench))
        labels = (
            grp.get("well_name", grp["well_i"]).astype(str)
            + "<br>"
            + grp.get("first_prod_date", pd.Series("", index=grp.index)).astype(str)
        )
        fig.add_trace(go.Scatter(
            x=grp[x_col],
            y=grp["elevation_i"],
            mode="markers+text",
            name=str(bench),
            text=labels,
            textposition="top center",
            textfont=dict(size=9),
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
) -> None:
    """
    Layer 2 — right-triangle zigzag connectors between adjacent well pairs.

    For each adjacent pair (wi, wk) in GB sorted order:
      - Horizontal segment:  wi → (wk.x, wi.y)
      - Vertical segment:    (wk.x, wi.y) → wk
      - Hypotenuse (dashed): wi → wk
      - Annotation at hypotenuse midpoint: horizontal_dist / vertical_dist / dist3d
    """
    dist3d_col = "3D_dist" if "3D_dist" in IK.columns else "dist3d"

    for idx in range(len(GB) - 1):
        wi = GB.iloc[idx]
        wk = GB.iloc[idx + 1]

        x_wi, y_wi = wi[x_col], wi["elevation_i"]
        x_wk, y_wk = wk[x_col], wk["elevation_i"]
        corner_x, corner_y = x_wk, y_wi

        # Horizontal segment
        fig.add_trace(go.Scatter(
            x=[x_wi, corner_x], y=[y_wi, corner_y],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Vertical segment
        fig.add_trace(go.Scatter(
            x=[corner_x, x_wk], y=[corner_y, y_wk],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Hypotenuse
        fig.add_trace(go.Scatter(
            x=[x_wi, x_wk], y=[y_wi, y_wk],
            mode="lines",
            line=dict(color="lightgray", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Spacing label from IK pairs
        pair = IK[
            (IK["well_i"] == wi["well_i"]) & (IK["well_k"] == wk["well_i"])
        ]
        if pair.empty:
            # Try reversed (IK may store both directions or only one)
            pair = IK[
                (IK["well_i"] == wk["well_i"]) & (IK["well_k"] == wi["well_i"])
            ]

        if not pair.empty:
            h_dist = pair["horizontal_dist"].iloc[0]
            v_dist = pair["vertical_dist"].iloc[0]
            d3d    = pair[dist3d_col].iloc[0]
            fig.add_annotation(
                x=(x_wi + x_wk) / 2,
                y=(y_wi + y_wk) / 2,
                text=(
                    f"H: {h_dist:,.0f} ft<br>"
                    f"V: {v_dist:,.0f} ft<br>"
                    f"3D: {d3d:,.0f} ft"
                ),
                showarrow=False,
                font=dict(size=8, color="dimgray"),
                bgcolor="rgba(255,255,255,0.75)",
                borderpad=2,
            )


def _add_formation_tops(
    fig: go.Figure,
    df_formation_tops: pd.DataFrame,
    GB: pd.DataFrame,
    x_col: str,
) -> None:
    """
    Layer 3 (optional) — dashed formation horizon lines across the GB x range.

    df_formation_tops columns: uwi, formation, top_tvd (ft, negative convention)
    """
    FORMATION_COLORS = {
        "FORM A1": "#e41a1c",
        "FORM B1": "#377eb8",
        "FORM C1": "#4daf4a",
    }
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

        color = FORMATION_COLORS.get(str(formation).upper(), "#888888")
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=str(formation),
            line=dict(color=color, width=1.5, dash="longdash"),
            hovertemplate=f"{formation}: %{{y:,.0f}} ft TVD<extra></extra>",
        ))


def build_gun_barrel_figure(
    GB: pd.DataFrame,
    IK: pd.DataFrame,
    x_col: str = "sectionDist",
    df_formation_tops: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Build the enhanced gun barrel figure.

    Args:
        GB: Output of compute_gun_barrel() — must contain x_col, elevation_i, bench.
        IK: Filtered IK spacing pairs — must contain horizontal_dist, vertical_dist, dist3d/3D_dist.
        x_col: 'sectionDist' (centered) or 'cum_dist' (from reference).
        df_formation_tops: Optional formation top data; omit layer if None.

    Returns:
        Plotly Figure with up to three trace layers.
    """
    if GB.empty:
        return empty_figure("No gun barrel data available.")

    fig = go.Figure()

    _add_well_points(fig, GB, x_col)
    _add_spacing_zigzag(fig, GB, IK, x_col)

    if df_formation_tops is not None and not df_formation_tops.empty:
        _add_formation_tops(fig, df_formation_tops, GB, x_col)

    x_label = "Section Distance (ft)" if x_col == "sectionDist" else "Cumulative Distance (ft)"

    fig.update_layout(
        title="Gun Barrel — Cross-Section View",
        xaxis=dict(
            title=x_label,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title="Depth TVD (ft)",
            autorange="reversed",
        ),
        legend=dict(title="Bench / Formation", orientation="h", y=-0.15),
        hovermode="closest",
        template="plotly_white",
        margin=dict(t=50, b=80, l=60, r=20),
    )
    return fig
