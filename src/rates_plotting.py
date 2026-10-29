"""Plotly figures for the rates, pricing and portfolio pages.

Colour roles (validated for colour-vision deficiency on the light theme):
blue = Vietnam / main model, amber = United States / second model, teal = third series,
grey = benchmark (random walk) or reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, AMBER, TEAL, GREY, INK = "#2563eb", "#d97706", "#0d9488", "#6b7280", "#1f2937"
MODEL_COLORS = {"Random walk": GREY, "DNS-AR(1)": BLUE, "AR(1) on yields": AMBER}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, y_title: str, height: int = 400, x_title: str | None = None,
            y_fmt: str = ".2f") -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      yaxis_title=y_title, xaxis_title=x_title, hovermode="closest")
    fig.update_yaxes(tickformat=y_fmt, gridcolor="#eef0f3", zerolinecolor="#d1d5db")
    fig.update_xaxes(gridcolor="#eef0f3")
    return fig


def _tenor_axis(fig: go.Figure, tenors) -> go.Figure:
    fig.update_xaxes(type="log", tickvals=list(tenors),
                     ticktext=[f"{t:g}Y" if t >= 1 else f"{round(t * 12)}M" for t in tenors],
                     title="Maturity (log scale)")
    return fig


def yield_history_figure(yields: pd.DataFrame, tenors=(1.0, 5.0, 10.0)) -> go.Figure:
    fig = go.Figure()
    for t, c in zip(tenors, (TEAL, BLUE, AMBER)):
        if t in yields:
            fig.add_trace(go.Scatter(x=yields.index, y=yields[t], name=f"{t:g}Y", mode="lines",
                                     line=dict(color=c, width=2), hovertemplate="%{y:.2f}%"))
    fig = _layout(fig, "Yield (%)", 340)
    fig.update_layout(hovermode="x unified")
    return fig


def curve_snapshot_figure(curves: list[tuple[str, str, np.ndarray, np.ndarray, object]]) -> go.Figure:
    """Published par yields (markers) and the calibrated curve's par yields (line).

    ``curves`` items: (label, colour, maturities, observed %, ZeroCurve).
    """
    fig = go.Figure()
    all_t = []
    for label, color, mats, obs, curve in curves:
        grid = np.geomspace(max(min(mats), 1 / 12), max(mats), 80)
        fig.add_trace(go.Scatter(x=grid, y=curve.par_yield(grid), name=f"{label} fitted ({curve.model})",
                                 line=dict(color=color, width=2), hovertemplate="%{x:.2f}y: %{y:.2f}%"))
        fig.add_trace(go.Scatter(x=mats, y=obs, name=f"{label} published", mode="markers",
                                 marker=dict(color=color, size=9, line=dict(color="white", width=2)),
                                 hovertemplate="%{x:g}y: %{y:.3f}%"))
        all_t += list(mats)
    fig = _layout(fig, "Par yield (%)", 380)
    ticks = sorted({t for t in all_t if t >= 1} | {0.25, 0.5})
    return _tenor_axis(fig, [t for t in ticks if t >= min(all_t)])


def curve_forecast_figure(current: pd.Series, fc: pd.DataFrame, model: str, h: int) -> go.Figure:
    g = fc[fc["model"] == model]
    rw = fc[fc["model"] == "Random walk"]
    c = MODEL_COLORS.get(model, BLUE)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["tenor"], y=g["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["tenor"], y=g["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=_rgba(c, 0.14), name="Prediction band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=current.index, y=current.values, name=f"Latest ({current.name:%b %Y})",
                             mode="lines+markers", line=dict(color=INK, width=2), marker=dict(size=8),
                             hovertemplate="%{y:.2f}%"))
    if model != "Random walk":
        fig.add_trace(go.Scatter(x=rw["tenor"], y=rw["point"], name="Random walk (no change)",
                                 line=dict(color=GREY, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["tenor"], y=g["point"], name=f"{model}, t + {h}m", mode="lines+markers",
                             line=dict(color=c, width=2), marker=dict(size=8), hovertemplate="%{y:.2f}%"))
    return _tenor_axis(_layout(fig, "Yield (%)", 380), current.index)


def rel_rmse_by_tenor_figure(summ: pd.DataFrame, h: int) -> go.Figure:
    s = summ[(summ["h"] == h) & (summ["model"] != "Random walk")]
    fig = go.Figure()
    for model, g in s.groupby("model"):
        sig = g["DM_pvalue"] < 0.05
        fig.add_trace(go.Scatter(
            x=g["tenor"], y=g["rel_RMSE"], name=model, mode="lines+markers",
            line=dict(color=MODEL_COLORS[model], width=2),
            marker=dict(size=np.where(sig, 11, 8), symbol=np.where(sig, "diamond", "circle"),
                        line=dict(color="white", width=2)),
            customdata=np.column_stack([g["DM_pvalue"], g["RMSE_bp"]]),
            hovertemplate="%{x:g}Y: %{y:.2f}× RW, RMSE %{customdata[1]:.0f} bp, DM p=%{customdata[0]:.3f}"))
    fig.add_hline(y=1, line=dict(color=GREY, dash="dash"), annotation_text="random walk",
                  annotation_position="bottom right")
    fig = _layout(fig, "RMSE ÷ random-walk RMSE", 340)
    return _tenor_axis(fig, sorted(s["tenor"].unique()))


def frontier_figure(frontier: pd.DataFrame, universe_tbl: pd.DataFrame, chosen: dict, vols: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier["vol_pct"], y=frontier["exp_return_pct"], name="Efficient frontier",
                             mode="lines", line=dict(color=BLUE, width=2),
                             customdata=frontier["duration"],
                             hovertemplate="risk %{x:.2f}%, carry+roll %{y:.2f}%, duration %{customdata:.1f}"))
    fig.add_trace(go.Scatter(x=vols, y=universe_tbl["exp_return_pct"], name="Single bonds", mode="markers+text",
                             text=[f"{t:g}Y" for t in universe_tbl.index], textposition="top center",
                             marker=dict(color=GREY, size=8), textfont=dict(color="#4b5563", size=11),
                             hovertemplate="%{text}: risk %{x:.2f}%, carry+roll %{y:.2f}%"))
    fig.add_trace(go.Scatter(x=[chosen["vol_pct"]], y=[chosen["exp_return_pct"]], name="Selected portfolio",
                             mode="markers", marker=dict(color=AMBER, size=14, symbol="diamond",
                                                         line=dict(color="white", width=2)),
                             hovertemplate="risk %{x:.2f}%, carry+roll %{y:.2f}%"))
    return _layout(fig, "Expected carry + roll-down (% over horizon)", 400,
                   "Risk: P&L volatility over horizon (%)")


def bar_figure(labels, values, y_title: str, color: str = BLUE, fmt: str = ".2f", height: int = 300) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=color, line=dict(color="white", width=2)),
                           hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>"))
    return _layout(fig, y_title, height, y_fmt=fmt)


def forward_points_figure(df: pd.DataFrame) -> go.Figure:
    """Forward points (VND) by tenor under each rate source."""
    fig = go.Figure()
    for col, color in zip([c for c in df.columns if c != "tenor_label"], (BLUE, AMBER, TEAL)):
        fig.add_trace(go.Scatter(x=df["tenor_label"], y=df[col], name=col, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=8),
                                 hovertemplate="%{x}: %{y:+,.0f} VND"))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    return _layout(fig, "Forward points (VND per USD)", 340, y_fmt=",.0f")


def option_value_figure(spots, values, payoff, spot_now: float, kind: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=payoff, name="Payoff at expiry", line=dict(color=GREY, width=2, dash="dash"),
                             hovertemplate="S=%{x:,.0f}: %{y:,.0f}"))
    fig.add_trace(go.Scatter(x=spots, y=values, name=f"{kind.title()} value today", line=dict(color=BLUE, width=2),
                             hovertemplate="S=%{x:,.0f}: %{y:,.1f}"))
    fig.add_vline(x=spot_now, line=dict(color=INK, dash="dot"), annotation_text="spot",
                  annotation_position="top")
    fig = _layout(fig, "VND per 1 USD notional", 340, "USD/VND spot", y_fmt=",.0f")
    fig.update_xaxes(tickformat=",.0f")
    return fig
