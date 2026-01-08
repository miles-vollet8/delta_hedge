import pandas as pd
import plotly.graph_objects as go

############################
# Plot your P&L
#############################
def plot_single_contract_pnl( # my localhost doesnt work for plotly so generated this with ai, should work just pass in the pnl_df
    df: pd.DataFrame,
    *,
    ret_col: str = "total_return",
    pnl_col: str = "position_pnl",
    base_col: str = "option_value",
    ts_col: str = "timestamp",
    title: str = "Hedged P&L vs Cumulative Stock Return"
) -> go.Figure:
    """
    Hedged Profit chart:
      - Left axis: total_return (%) + position_pnl as % of initial option_value
      - Right axis: position_pnl (absolute)
    """
    # x-axis
    if ts_col in df.columns:
        x = pd.to_datetime(df[ts_col])
    else:
        x = df.index

    for col in (ret_col, pnl_col, base_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in df")

    # base notional: first option_value
    base_val = df[base_col].iloc[0]
    if base_val == 0:
        raise ValueError("First option_value is zero; cannot normalize P&L to %.")

    # stock return in percent 
    stock_ret_pct = df[ret_col] * 100.0

    # position P&L as % of initial option value
    pnl_pct = df[pnl_col] / base_val * 100.0

    # absolute P&L
    pnl_abs = df[pnl_col]

    fig = go.Figure()

    # 1) Cumulative stock return (%)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=stock_ret_pct,
            mode="lines",
            name="Cum stock return (%)",
            yaxis="y1",
        )
    )

    # 2) Position P&L as % of initial option value
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pnl_pct,
            mode="lines",
            name="Position P&L (% of premium)",
            yaxis="y1",
        )
    )

    # 3) Position P&L in absolute terms on secondary axis
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pnl_abs,
            mode="lines",
            name="Position P&L ($)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Returns / P&L (%)",
            tickformat=".1f",
            zeroline=True,
            zerolinewidth=1,
        ),
        yaxis2=dict(
            title="Position P&L (absolute)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
        ),
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
        ),
        margin=dict(l=70, r=70, t=40, b=40),
    )
    return fig


def plot_portfolio_pnl( #same here just pass in portfolio df and your initial capital since entry and exit gets tricky for returns in percent terms
    df: pd.DataFrame,
    *,
    initial_capital: float,
    ret_col: str = "total_return",
    pnl_col: str = "position_pnl",
    ts_col: str = "timestamp",
    title: str = "Hedged P&L vs Cumulative Stock Return (Capital-Based %)"
) -> go.Figure:
    """
    Hedged Profit chart:
      - Left axis: stock total_return (%) + position_pnl as % of initial_capital
      - Right axis: position_pnl (absolute)
    """
    if initial_capital is None or initial_capital == 0:
        raise ValueError("initial_capital must be non-zero.")

    # x-axis
    if ts_col in df.columns:
        x = pd.to_datetime(df[ts_col])
    else:
        x = df.index

    for col in (ret_col, pnl_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in df")

    # already % as decimal → convert to percent
    stock_ret_pct = df[ret_col] * 100.0

    # PnL as % of user-specified capital
    pnl_pct = df[pnl_col] / initial_capital * 100.0

    # absolute PnL
    pnl_abs = df[pnl_col]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=stock_ret_pct,
            mode="lines",
            name="Cum stock return (%)",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pnl_pct,
            mode="lines",
            name="Position P&L (% of capital)",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pnl_abs,
            mode="lines",
            name="Position P&L ($)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Returns / P&L (%)",
            tickformat=".1f",
            zeroline=True,
        ),
        yaxis2=dict(
            title="Position P&L (absolute)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
        ),
        legend=dict(orientation="h", x=0.0, y=1.02),
        margin=dict(l=70, r=70, t=40, b=40),
    )

    return fig


