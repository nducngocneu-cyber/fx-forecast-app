"""Plotly figures used by the Streamlit app."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "History": "#1f2937",
    "Random walk": "#6b7280",
    "ARIMA": "#2563eb",
    "Gradient boosting": "#d97706",
}
_BAND_ALPHA = 0.14


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, y_title: str, height: int = 460) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=y_title, xaxis_title=None,
    )
    fig.update_yaxes(tickformat=",.0f", gridcolor="#eef0f3")
    fig.update_xaxes(gridcolor="#eef0f3")
    return fig


def forecast_figure(history: pd.Series, forecasts: dict[str, pd.DataFrame],
                    lookback: int, level: float) -> go.Figure:
    """History (last ``lookback`` days) plus each model's point forecast and band."""
    hist = history.iloc[-lookback:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="USD/VND (close)",
                             line=dict(color=COLORS["History"], width=1.6),
                             hovertemplate="%{y:,.0f}"))
    anchor_x, anchor_y = hist.index[-1], hist.iloc[-1]

    for name, fc in forecasts.items():
        c = COLORS.get(name, "#10b981")
        x = [anchor_x, *fc.index]
        fig.add_trace(go.Scatter(x=x, y=[anchor_y, *fc["upper"]], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", legendgroup=name))
        fig.add_trace(go.Scatter(x=x, y=[anchor_y, *fc["lower"]], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(c, _BAND_ALPHA),
                                 name=f"{name} {level:.0%} band", legendgroup=name,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[anchor_y, *fc["point"]], name=name, legendgroup=name,
                                 line=dict(color=c, width=2, dash="dash" if name == "Random walk" else "solid"),
                                 hovertemplate="%{y:,.0f}"))
    return _layout(fig, "VND per USD")


def history_figure(raw: pd.Series, cleaned: pd.Series) -> go.Figure:
    """Raw feed vs cleaned series, to make the data-quality step visible."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=raw.values, name="Raw feed",
                             line=dict(color="#f87171", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=cleaned.index, y=cleaned.values, name="Cleaned",
                             line=dict(color=COLORS["History"], width=1.3)))
    return _layout(fig, "VND per USD", height=360)


def rmse_by_horizon_figure(summary: pd.DataFrame) -> go.Figure:
    """RMSE relative to the random walk (1.0 = same accuracy) by horizon."""
    fig = go.Figure()
    for name, g in summary.groupby("model"):
        fig.add_trace(go.Scatter(
            x=g["h"], y=g["rel_RMSE"], name=name, mode="lines+markers", marker=dict(size=4),
            line=dict(color=COLORS.get(name, "#10b981"), width=2,
                      dash="dash" if name == "Random walk" else "solid"),
            hovertemplate="h=%{x}: %{y:.3f}×",
        ))
    fig = _layout(fig, "RMSE ÷ random-walk RMSE", height=380)
    fig.update_yaxes(tickformat=".2f")
    fig.update_xaxes(title="Forecast horizon (business days)")
    return fig


def coverage_figure(summary: pd.DataFrame, level: float) -> go.Figure:
    fig = go.Figure()
    for name, g in summary.groupby("model"):
        fig.add_trace(go.Scatter(x=g["h"], y=g["coverage"], name=name, mode="lines+markers",
                                 marker=dict(size=4), line=dict(color=COLORS.get(name), width=2),
                                 hovertemplate="h=%{x}: %{y:.0%}"))
    fig.add_hline(y=level, line=dict(color="#9ca3af", dash="dot"),
                  annotation_text=f"nominal {level:.0%}", annotation_position="bottom right")
    fig = _layout(fig, "Share of actuals inside band", height=340)
    fig.update_yaxes(tickformat=".0%", range=[0.5, 1.02])
    fig.update_xaxes(title="Forecast horizon (business days)")
    return fig
