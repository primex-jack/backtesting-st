# supertrend_chart.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_supertrend_chart(df, trades, date_start, date_end, symbol, timeframe, output_filename="supertrend_chart.html"):
    """
    Plot a candlestick chart with Supertrend line and LONG/SHORT signals using Plotly.
    The Supertrend line is a single continuous line, colored green for LONG (trend == 1)
    and red for SHORT (trend == -1), switching colors at trend transitions.

    Parameters:
    - df: DataFrame with OHLC data, line_st (ST_LINE), and trend (from backtest_strategy)
    - trades: List of trade dictionaries (from backtest_strategy)
    - date_start: Start date for the chart (string or datetime)
    - date_end: End date for the chart (string or datetime)
    - symbol: Trading pair (e.g., BTCUSDT)
    - timeframe: Chart timeframe (e.g., 3m)
    - output_filename: Name of the output HTML file
    """
    # Filter DataFrame for the specified date range
    df = df[(df.index >= date_start) & (df.index <= date_end)].copy()

    # Create the candlestick chart
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))

    # Segment the Supertrend line based on trend changes
    segments = []
    current_segment = {'x': [], 'y': [], 'trend': None}

    for i in range(len(df)):
        timestamp = df.index[i]
        line_st_value = df['line_st'].iloc[i]
        trend_value = df['trend'].iloc[i]

        # Skip if line_st or trend is NaN
        if pd.isna(line_st_value) or pd.isna(trend_value):
            continue

        # Start a new segment if trend changes or it's the first point
        if i == 0 or trend_value != current_segment['trend']:
            if current_segment['x']:  # Save the previous segment if it exists
                segments.append({
                    'x': current_segment['x'],
                    'y': current_segment['y'],
                    'color': 'green' if current_segment['trend'] == 1 else 'red',
                    'name': 'Supertrend LONG' if current_segment['trend'] == 1 else 'Supertrend SHORT'
                })
            # Start a new segment
            current_segment = {'x': [timestamp], 'y': [line_st_value], 'trend': trend_value}
        else:
            # Continue the current segment
            current_segment['x'].append(timestamp)
            current_segment['y'].append(line_st_value)

    # Add the last segment
    if current_segment['x']:
        segments.append({
            'x': current_segment['x'],
            'y': current_segment['y'],
            'color': 'green' if current_segment['trend'] == 1 else 'red',
            'name': 'Supertrend LONG' if current_segment['trend'] == 1 else 'Supertrend SHORT'
        })

    # Plot each segment as a separate Scatter trace
    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            line=dict(color=segment['color'], width=2),
            name=segment['name'],
            showlegend=False  # Avoid duplicate legend entries; we'll handle it manually
        ))

    # Extract LONG and SHORT signals from trades
    long_entries = []
    long_prices = []
    short_entries = []
    short_prices = []

    for trade in trades:
        if trade['type'] == 'entry':
            timestamp = trade['timestamp']
            price = trade['price']
            if trade['side'] == 'long':
                long_entries.append(timestamp)
                long_prices.append(price)
            elif trade['side'] == 'short':
                short_entries.append(timestamp)
                short_prices.append(price)

    # Add LONG signals (green upward triangles)
    if long_entries:
        fig.add_trace(go.Scatter(
            x=long_entries,
            y=long_prices,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='LONG Entry'
        ))

    # Add SHORT signals (red downward triangles)
    if short_entries:
        fig.add_trace(go.Scatter(
            x=short_entries,
            y=short_prices,
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='SHORT Entry'
        ))

    # Update layout
    fig.update_layout(
        title=f'{symbol} Supertrend Chart ({timeframe}) - {date_start} to {date_end}',
        yaxis_title='Price (USDT)',
        xaxis_title='Date',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        # Add legend entries for Supertrend lines
        legend=dict(
            traceorder='normal',
            itemsizing='constant'
        )
    )

    # Save the chart as an HTML file
    fig.write_html(output_filename)
    print(f"Chart saved as {output_filename}")

    # Optionally, display the chart in the default browser
    # fig.show()