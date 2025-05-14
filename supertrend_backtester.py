import sys
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime, timedelta
from binance_history_loader import load_or_update_klines, Client
import itertools
import concurrent.futures
from tqdm import tqdm
import signal
import os
import json

# Import the charting function
from supertrend_chart import plot_supertrend_chart

# Script Version
SCRIPT_VERSION = "2.10.9"

# Set up main logger with INFO level for main process only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.propagate = False

# Global stop flag for multiprocessing
stop_flag = False


def calculate_atr(df, period, quiet=False):
    if not quiet:
        print(f"Calculating ATR with period {period}")
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=df.index)
    atr.iloc[period - 1] = true_range.iloc[:period].mean()
    for i in range(period, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + true_range.iloc[i]) / period
    if not quiet:
        print("ATR calculation completed")
    return atr


def calculate_ema(series, period, quiet=False):
    if not quiet:
        print(f"Calculating EMA with period {period}")
    alpha = 2 / (period + 1)
    ema = series.ewm(alpha=alpha, adjust=False).mean()
    if not quiet:
        print("EMA calculation completed")
    return ema


def calculate_supertrend(df, atr_period, atr_ratio, quiet=False):
    if not quiet:
        print("Starting Supertrend calculation")
    atr = calculate_atr(df, atr_period, quiet)
    atr_smma = calculate_ema(atr, atr_period, quiet)
    delta_stop = atr_smma * atr_ratio
    up = df['close'] - delta_stop
    dn = df['close'] + delta_stop
    trend_up = pd.Series(0.0, index=df.index)
    trend_down = pd.Series(0.0, index=df.index)
    trend = pd.Series(0.0, index=df.index)
    trend.iloc[0] = 1
    trend_up.iloc[0] = up.iloc[0]
    trend_down.iloc[0] = dn.iloc[0]
    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] > trend_up.iloc[i - 1]:
            trend_up.iloc[i] = max(up.iloc[i], trend_up.iloc[i - 1])
        else:
            trend_up.iloc[i] = up.iloc[i]
        if df['close'].iloc[i - 1] < trend_down.iloc[i - 1]:
            trend_down.iloc[i] = min(dn.iloc[i], trend_down.iloc[i - 1])
        else:
            trend_down.iloc[i] = dn.iloc[i]
        line_st = pd.Series(np.where(trend.iloc[i - 1] == 1, trend_up, trend_down), index=df.index)
        if df['close'].iloc[i] > line_st.iloc[i]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i] < line_st.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    line_st = pd.Series(np.where(trend == 1, trend_up, trend_down), index=df.index)
    if not quiet:
        print("Supertrend calculation completed")
    return line_st, trend


def backtest_strategy(df, atr_period, atr_ratio, start_date, end_date, verbose=False, quiet=False):
    if not quiet:
        print(f"Starting backtest_strategy with df type: {type(df)}, atr_period: {atr_period}, atr_ratio: {atr_ratio}")
    line_st, trend = calculate_supertrend(df, atr_period, atr_ratio, quiet)
    if not quiet:
        print("Adding Supertrend columns to DataFrame")
    df['line_st'] = line_st
    df['prev_line_st'] = df['line_st'].shift(1)
    df['trend'] = trend
    if not quiet:
        print("Filtering DataFrame for date range")
    df_in_range = df[(df.index >= start_date) & (df.index <= end_date)]
    last_close = df_in_range['close'].iloc[-1] if not df_in_range.empty else df['close'].iloc[-1]
    if not quiet:
        print(f"Last close: {last_close}")

    position = 0
    last_entry_price = 0.0
    last_entry_timestamp = None
    trades = []
    total_profit = 0.0

    global stop_flag
    for i in range(1, len(df)):
        if stop_flag:
            if not quiet:
                print("Stop flag detected, exiting backtest_strategy")
            break
        timestamp = df.index[i]
        close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i - 1]
        line_st_curr = df['line_st'].iloc[i]
        line_st_prev = df['prev_line_st'].iloc[i]
        trend_curr = df['trend'].iloc[i]
        if start_date <= timestamp <= end_date:
            if verbose:
                print(
                    f"Timestamp {timestamp}: Close={close:.2f}, Line_ST={line_st_curr:.2f}, "
                    f"Prev Line_ST={line_st_prev:.2f}, Trend={trend_curr:.1f}, "
                    f"Current Position={position}, Last Entry Price={last_entry_price:.2f}"
                )
            long_condition = close >= line_st_curr and prev_close < line_st_prev and trend_curr == 1
            short_condition = close <= line_st_curr and prev_close > line_st_prev and trend_curr == -1
            if position == 0:
                if long_condition:
                    position = 1
                    last_entry_price = close
                    last_entry_timestamp = timestamp
                    if not quiet:
                        print(f"Long entry at {close:.2f} at {timestamp}")
                    trades.append({
                        'type': 'entry',
                        'side': 'long',
                        'price': close,
                        'timestamp': timestamp,
                        'max_runup': 0.0,
                        'max_drawdown': 0.0
                    })
                elif short_condition:
                    position = -1
                    last_entry_price = close
                    last_entry_timestamp = timestamp
                    if not quiet:
                        print(f"Short entry at {close:.2f} at {timestamp}")
                    trades.append({
                        'type': 'entry',
                        'side': 'short',
                        'price': close,
                        'timestamp': timestamp,
                        'max_runup': 0.0,
                        'max_drawdown': 0.0
                    })
            elif position == 1:
                if short_condition:
                    if not quiet:
                        print(f"Calculating max run-up for long trade from {last_entry_timestamp} to {timestamp}")
                    trade_period = df[(df.index >= last_entry_timestamp) & (df.index <= timestamp)]
                    if trade_period.empty:
                        if not quiet:
                            print(f"Warning: trade_period is empty for {last_entry_timestamp} to {timestamp}")
                        max_runup = 0.0
                        max_drawdown = 0.0
                    else:
                        max_runup = (trade_period['high'] - last_entry_price).max()
                        lowest_price = trade_period['low'].min()
                        max_drawdown = max(last_entry_price - lowest_price, 0)
                        if not quiet:
                            print(f"Lowest price during trade: {lowest_price} at {trade_period['low'].idxmin()}")
                            print(f"Trade period range: {trade_period.index[0]} to {trade_period.index[-1]}")
                    profit = close - last_entry_price
                    total_profit += profit
                    if not quiet:
                        print(f"Long exit at {close:.2f}, Profit: {profit:.2f} at {timestamp}")
                    trades.append({
                        'type': 'exit',
                        'side': 'long',
                        'entry_price': last_entry_price,
                        'entry_timestamp': last_entry_timestamp,
                        'price': close,
                        'profit': profit,
                        'max_runup': max_runup if not pd.isna(max_runup) else 0.0,
                        'max_drawdown': max_drawdown if not pd.isna(max_drawdown) else 0.0,
                        'timestamp': timestamp
                    })
                    position = -1
                    last_entry_price = close
                    last_entry_timestamp = timestamp
                    if not quiet:
                        print(f"Short entry at {close:.2f} at {timestamp}")
                    trades.append({
                        'type': 'entry',
                        'side': 'short',
                        'price': close,
                        'timestamp': timestamp,
                        'max_runup': 0.0,
                        'max_drawdown': 0.0
                    })
            elif position == -1:
                if long_condition:
                    if not quiet:
                        print(f"Calculating max run-up for short trade from {last_entry_timestamp} to {timestamp}")
                    trade_period = df[(df.index >= last_entry_timestamp) & (df.index <= timestamp)]
                    if trade_period.empty:
                        if not quiet:
                            print(f"Warning: trade_period is empty for {last_entry_timestamp} to {timestamp}")
                        max_runup = 0.0
                        max_drawdown = 0.0
                    else:
                        max_runup = (last_entry_price - trade_period['low']).max()
                        max_drawdown = (trade_period['high'] - last_entry_price).max()
                        if not quiet:
                            print(f"Highest price during trade: {trade_period['high'].max()} at {trade_period['high'].idxmax()}")
                            print(f"Trade period range: {trade_period.index[0]} to {trade_period.index[-1]}")
                    profit = last_entry_price - close
                    total_profit += profit
                    if not quiet:
                        print(f"Short exit at {close:.2f}, Profit: {profit:.2f} at {timestamp}")
                    trades.append({
                        'type': 'exit',
                        'side': 'short',
                        'entry_price': last_entry_price,
                        'entry_timestamp': last_entry_timestamp,
                        'price': close,
                        'profit': profit,
                        'max_runup': max_runup if not pd.isna(max_runup) else 0.0,
                        'max_drawdown': max_drawdown if not pd.isna(max_drawdown) else 0.0,
                        'timestamp': timestamp
                    })
                    position = 1
                    last_entry_price = close
                    last_entry_timestamp = timestamp
                    if not quiet:
                        print(f"Long entry at {close:.2f} at {timestamp}")
                    trades.append({
                        'type': 'entry',
                        'side': 'long',
                        'price': close,
                        'timestamp': timestamp,
                        'max_runup': 0.0,
                        'max_drawdown': 0.0
                    })

    if position != 0:
        if not quiet:
            print(f"Calculating max run-up for open trade from {last_entry_timestamp}")
        trade_period = df[(df.index >= last_entry_timestamp)]
        if position == 1:
            max_runup = (trade_period['high'] - last_entry_price).max()
            max_drawdown = max(last_entry_price - trade_period['low'].min(), 0)
            if not quiet:
                print(f"Lowest price during open trade: {trade_period['low'].min()} at {trade_period['low'].idxmin()}")
        else:
            max_runup = (last_entry_price - trade_period['low']).max()
            max_drawdown = (trade_period['high'] - last_entry_price).max()
            if not quiet:
                print(f"Highest price during open trade: {trade_period['high'].max()} at {trade_period['high'].idxmax()}")
        unrealized_pl = last_close - last_entry_price if position == 1 else last_entry_price - last_close
        trades.append({
            'type': 'open',
            'side': 'long' if position == 1 else 'short',
            'entry_price': last_entry_price,
            'entry_timestamp': last_entry_timestamp,
            'exit_price': last_close,
            'unrealized_pl': unrealized_pl,
            'max_runup': max_runup if not pd.isna(max_runup) else 0.0,
            'max_drawdown': max_drawdown if not pd.isna(max_drawdown) else 0.0,
            'timestamp': df.index[i - 1]
        })

    if not quiet:
        print("backtest_strategy completed")
    return trades, total_profit, df


def compute_metrics(trades, total_profit, quiet=False):
    if not quiet:
        print("Starting compute_metrics")
    completed_trades = [trade for trade in trades if trade['type'] == 'exit']
    open_trades = [trade for trade in trades if trade['type'] == 'open']

    num_trades = len(completed_trades)
    winning_trades = sum(1 for trade in completed_trades if trade['profit'] > 0)
    losing_trades = sum(1 for trade in completed_trades if trade['profit'] <= 0)
    percent_profitable = (winning_trades / num_trades * 100) if num_trades > 0 else 0.0
    profits = [trade['profit'] for trade in completed_trades]
    largest_winning_trade = max(profits) if any(p > 0 for p in profits) else 0.0
    largest_losing_trade = min(profits) if any(p <= 0 for p in profits) else 0.0
    max_runup = max([trade['max_runup'] for trade in completed_trades], default=0.0)
    max_drawdown = max([trade['max_drawdown'] for trade in completed_trades], default=0.0)
    unrealized_pl = sum(trade['unrealized_pl'] for trade in open_trades) if open_trades else 0.0

    if not quiet:
        print("compute_metrics completed")
    return {
        'num_trades': num_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'percent_profitable': percent_profitable,
        'largest_winning_trade': largest_winning_trade,
        'largest_losing_trade': largest_losing_trade,
        'max_drawdown': max_drawdown,
        'max_runup': max_runup,
        'total_profit': total_profit,
        'unrealized_pl': unrealized_pl,
        'trades': trades
    }


def print_trade_summary(trades, total_profit, symbol, timeframe, date_start, date_end, atr_period, atr_ratio, starting_capital):
    metrics = compute_metrics(trades, total_profit)
    print("=" * 80)
    print("Detailed Trade List")
    print("=" * 80)
    cumulative_profit = 0.0
    for trade in trades:
        if trade['type'] == 'exit':
            cumulative_profit += trade['profit']
            side_label = "LONG" if trade['side'] == 'long' else "SHORT"
            print(
                f"{side_label:<5} | {trade['entry_timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<19} | "
                f"Entry: $ {trade['entry_price']:>10.2f} | "
                f"Exit: $ {trade['price']:>10.2f} | {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<19} | "
                f"Profit: $ {trade['profit']:>10.2f} | "
                f"Max RunUp: $ {trade['max_runup']:>10.2f} | "
                f"Max Drawdown: $ {trade['max_drawdown']:>10.2f} | "
                f"Cumulative Profit: $ {cumulative_profit:>10.2f}"
            )
        elif trade['type'] == 'open':
            print(
                f"{trade['entry_timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<19} | {trade['side'].capitalize():<5}  (Open) | "
                f"Entry: $ {trade['entry_price']:>10.2f} | "
                f"Last Price: $ {trade['exit_price']:>10.2f} | Unrealized P/L: $ {trade['unrealized_pl']:>10.2f} | "
                f"Max RunUp: $ {trade['max_runup']:>10.2f} | Max Drawdown: $ {trade['max_drawdown']:>10.2f}"
            )
    if completed_trades := [trade for trade in trades if trade['type'] == 'exit']:
        winning_profits = [trade['profit'] for trade in completed_trades if trade['profit'] > 0]
        losing_profits = [trade['profit'] for trade in completed_trades if trade['profit'] <= 0]
        avg_profit = np.mean(winning_profits) if winning_profits else 0.0
        avg_loss = np.mean(losing_profits) if losing_profits else 0.0
        avg_max_runup = np.mean([trade['max_runup'] for trade in completed_trades])
        avg_max_drawdown = np.mean([trade['max_drawdown'] for trade in completed_trades])
        print("=" * 80)
        print("AVERAGES (Completed Trades):")
        print(f"- AVG Profit:      $ {avg_profit:>10.2f}")
        print(f"- AVG Loss:        $ {avg_loss:>10.2f}")
        print(f"- AVG Max RunUp:   $ {avg_max_runup:>10.2f}")
        print(f"- AVG Max Drawdown: $ {avg_max_drawdown:>10.2f}")
    print("=" * 80)
    print()

    print("=" * 80)
    print("Backtest Parameters")
    print("=" * 80)
    print(f"Timeframe: {timeframe}")
    print(f"Trading Pair: {symbol}")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print(f"Date Range: {date_start} to {date_end}")
    print("=" * 80)
    print()

    print("Backtest Results")
    print("=" * 80)
    header = (
        "| ATR_PERIOD | ATR_RATIO | Trades | # Wins | # Losses | % Profitable | Total P/L  | Open P/L  | Biggest Profit  | Biggest Loss  | Max Drawdown  | Max Runup  |"
    )
    print(header)
    separator = (
        "|------------|-----------|--------|--------|----------|--------------|------------|-----------|-----------------|---------------|---------------|------------|"
    )
    print(separator)
    row = (
        f"| {atr_period:<10} | {atr_ratio:<9.1f} | {metrics['num_trades']:>6} | {metrics['winning_trades']:>6} | {metrics['losing_trades']:>8} | "
        f"{metrics['percent_profitable']:>11.2f}% | ${metrics['total_profit']:>9.2f} | ${metrics['unrealized_pl']:>8.2f} | ${metrics['largest_winning_trade']:>14.2f} | "
        f"${metrics['largest_losing_trade']:>12.2f} | ${metrics['max_drawdown']:>12.2f} | ${metrics['max_runup']:>9.2f} |"
    )
    print(row)
    print("=" * 80)


def print_range_summary(results, symbol, timeframe, date_start, date_end, starting_capital, top_n=10):
    sorted_results = sorted([r for r in results if r is not None], key=lambda x: x[2]['total_profit'], reverse=True)
    top_results = sorted_results[:top_n]

    print("=" * 80)
    print("Backtest Parameters")
    print("=" * 80)
    print(f"Timeframe: {timeframe}")
    print(f"Trading Pair: {symbol}")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print(f"Date Range: {date_start} to {date_end}")
    print("=" * 80)
    print()

    print(f"Top {top_n} Backtest Results (Sorted by Total Profit)")
    print("=" * 80)
    header = (
        "| Rank | ATR_PERIOD | ATR_RATIO | Trades | # Wins | # Losses | % Profitable | Total P/L  | Open P/L  | Biggest Profit  | Biggest Loss  | Max Drawdown  | Max Runup  |"
    )
    print(header)
    separator = (
        "|------|------------|-----------|--------|--------|----------|--------------|------------|-----------|-----------------|---------------|---------------|------------|"
    )
    print(separator)
    for rank, (atr_period, atr_ratio, metrics) in enumerate(top_results, 1):
        row = (
            f"| {rank:<4} | {atr_period:<10} | {atr_ratio:<9.1f} | {metrics['num_trades']:>6} | {metrics['winning_trades']:>6} | {metrics['losing_trades']:>8} | "
            f"{metrics['percent_profitable']:>11.2f}% | ${metrics['total_profit']:>9.2f} | ${metrics['unrealized_pl']:>8.2f} | ${metrics['largest_winning_trade']:>14.2f} | "
            f"${metrics['largest_losing_trade']:>12.2f} | ${metrics['max_drawdown']:>12.2f} | ${metrics['max_runup']:>9.2f} |"
        )
        print(row)
    print("=" * 80)
    print()

    if top_results:
        best_atr_period, best_atr_ratio, best_metrics = top_results[0]
        print(f"Detailed Trade List for Best Performer (ATR_PERIOD={best_atr_period}, ATR_RATIO={best_atr_ratio})")
        print("=" * 80)
        cumulative_profit = 0.0
        for trade in best_metrics['trades']:
            if trade['type'] == 'exit':
                cumulative_profit += trade['profit']
                side_label = "LONG" if trade['side'] == 'long' else "SHORT"
                print(
                    f"{side_label:<5} | {trade['entry_timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<19} | "
                    f"Entry: $ {trade['entry_price']:>10.2f} | "
                    f"Exit: $ {trade['price']:>10.2f} | {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<19} | "
                    f"Profit: $ {trade['profit']:>10.2f} | "
                    f"Max RunUp: $ {trade['max_runup']:>10.2f} | "
                    f"Max Drawdown: $ {trade['max_drawdown']:>10.2f} | "
                    f"Cumulative Profit: $ {cumulative_profit:>10.2f}"
                )
            elif trade['type'] == 'open':
                print(
                    f"{trade['entry_timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<19} | {trade['side'].capitalize():<5}  (Open) | "
                    f"Entry: $ {trade['entry_price']:>10.2f} | "
                    f"Last Price: $ {trade['exit_price']:>10.2f} | Unrealized P/L: $ {trade['unrealized_pl']:>10.2f} | "
                    f"Max RunUp: $ {trade['max_runup']:>10.2f} | Max Drawdown: $ {trade['max_drawdown']:>10.2f}"
                )
        if completed_trades := [trade for trade in best_metrics['trades'] if trade['type'] == 'exit']:
            winning_profits = [trade['profit'] for trade in completed_trades if trade['profit'] > 0]
            losing_profits = [trade['profit'] for trade in completed_trades if trade['profit'] <= 0]
            avg_profit = np.mean(winning_profits) if winning_profits else 0.0
            avg_loss = np.mean(losing_profits) if losing_profits else 0.0
            avg_max_runup = np.mean([trade['max_runup'] for trade in completed_trades])
            avg_max_drawdown = np.mean([trade['max_drawdown'] for trade in completed_trades])
            print("=" * 80)
            print("AVERAGES (Completed Trades):")
            print(f"- AVG Profit:      $ {avg_profit:>10.2f}")
            print(f"- AVG Loss:        $ {avg_loss:>10.2f}")
            print(f"- AVG Max RunUp:   $ {avg_max_runup:>10.2f}")
            print(f"- AVG Max Drawdown: $ {avg_max_drawdown:>10.2f}")
        print("=" * 80)
        print("Backtest completed successfully. Exiting...")

def run_single_combination(df, atr_period, atr_ratio, start_date, end_date, quiet=False):
    if not quiet:
        print(f"Starting run_single_combination with atr_period: {atr_period}, atr_ratio: {atr_ratio}")
    trades, total_profit, df_with_indicators = backtest_strategy(df.copy(), atr_period, atr_ratio, start_date, end_date, quiet=quiet)
    if stop_flag:
        if not quiet:
            print("Stop flag detected in run_single_combination, returning None")
        return None, None, None
    metrics = compute_metrics(trades, total_profit, quiet)
    metrics['atr_period'] = atr_period
    metrics['atr_ratio'] = atr_ratio
    if not quiet:
        print(f"run_single_combination completed for atr_period: {atr_period}, atr_ratio: {atr_ratio}")
    return atr_period, atr_ratio, metrics

def main():
    parser = argparse.ArgumentParser(description='Supertrend Backtester')
    parser.add_argument('--symbol', default='ETHUSDT', help='Trading pair (e.g., ETHUSDT)')
    parser.add_argument('--timeframe', default='3m', help='Timeframe (e.g., 3m, 1h)')
    parser.add_argument('--date-start', default='2025-03-05', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date-end', default='2025-03-06', help='End date (YYYY-MM-DD)')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR Period (single mode)')
    parser.add_argument('--atr-ratio', type=float, default=6.0, help='ATR Ratio (single mode)')
    parser.add_argument('--atr-period-range', type=str, help='ATR Period range (range mode, e.g., 5-15)')
    parser.add_argument('--atr-ratio-range', type=str, help='ATR Ratio range (range mode, e.g., 5-25)')
    parser.add_argument('--starting-capital', type=float, default=10000.0, help='Starting capital (default: 10000.0)')
    parser.add_argument('--top-n', type=int, default=50, help='Number of top combinations to display (default: 50)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging to files in range mode')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes for range mode (default: 8)')
    parser.add_argument('--quiet', action='store_true', help='Disable detailed logging')
    parser.add_argument('--chart', action='store_true', help='Generate an interactive chart (single mode only)')

    args = parser.parse_args()

    print(f"Script Version: {SCRIPT_VERSION}")

    global stop_flag

    def signal_handler(sig, frame):
        print("\nReceived Ctrl+C, shutting down...")
        stop_flag = True

    signal.signal(signal.SIGINT, signal_handler)

    if args.verbose and not (args.atr_period_range and args.atr_ratio_range):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    start_date = pd.to_datetime(args.date_start)
    end_date = pd.to_datetime(args.date_end) + timedelta(days=1) - timedelta(seconds=1)

    # Cap end_date at today’s date to avoid requesting future data
    today = datetime.now()
    if end_date > today:
        logger.warning(f"Requested end date {end_date} is in the future. Capping at current date: {today}")
        end_date = today

    atr_period_for_warmup = args.atr_period
    if args.atr_period_range:
        atr_period_min = int(args.atr_period_range.split('-')[0])
        atr_period_for_warmup = atr_period_min

    timeframe_map = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
    minutes_per_bar = timeframe_map.get(args.timeframe, 3)

    bars_needed = 2 * atr_period_for_warmup
    extra_minutes = bars_needed * minutes_per_bar
    extended_start_date = start_date - timedelta(minutes=extra_minutes)
    extended_end_date = end_date + timedelta(days=1)
    extended_start_str = extended_start_date.strftime('%Y-%m-%d')
    end_str = extended_end_date.strftime('%Y-%m-%d')

    logger.info(f"Loading data for {args.symbol} from {extended_start_str} to {end_str}...")
    client = Client()
    df = load_or_update_klines(client, args.symbol, args.timeframe, extended_start_str, end_str)

    if df is None:
        logger.error("Failed to load data from Binance. Check API connectivity or data availability.")
        sys.exit(1)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif df.index.dtype != 'datetime64[ns]':
        raise ValueError("DataFrame index is not datetime. Check data loading.")

    if not args.quiet:
        print(f"Existing data range: {df.index[0]} ({df.index[0].value}) to {df.index[-1]} ({df.index[-1].value})")
        print(f"Requested data range: {start_date} ({int(start_date.timestamp() * 1000)}) to {end_date} ({int(end_date.timestamp() * 1000)})")
        print(f"Validating data: Data range from {df.index[0]} ({df.index[0].value}) to {df.index[-1]} ({df.index[-1].value})")
        print(f"Requested range: {start_date} ({int(start_date.timestamp() * 1000)}) to {end_date} ({int(end_date.timestamp() * 1000)})")

    logger.info(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    if args.atr_period_range and args.atr_ratio_range:
        logger.info("Running in range combination testing mode...")
        atr_period_range = [int(x) for x in args.atr_period_range.split('-')]
        atr_ratio_range = [float(x) for x in args.atr_ratio_range.split('-')]
        atr_periods = list(range(atr_period_range[0], atr_period_range[1] + 1))
        atr_ratios = list(np.arange(atr_ratio_range[0], atr_ratio_range[1] + 1, 1.0))
        combinations = list(itertools.product(atr_periods, atr_ratios))
        total_combinations = len(combinations)
        logger.info(f"Testing {total_combinations} combinations of ATR_PERIOD ({atr_period_range[0]}-{atr_period_range[1]}) and ATR_RATIO ({atr_ratio_range[0]}-{atr_ratio_range[1]})")

        if args.chart:
            print("Warning: Charting is only supported in single mode. Ignoring --chart parameter.")

        results = []

        file_handler = None
        if args.verbose:
            os.makedirs("logs", exist_ok=True)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        with tqdm(total=total_combinations, desc="Testing combinations", file=sys.stdout) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(run_single_combination, df.copy(), period, ratio, start_date, end_date, args.quiet)
                    for period, ratio in combinations
                ]
                for future in concurrent.futures.as_completed(futures):
                    if stop_flag:
                        executor.shutdown(wait=True)
                        break
                    try:
                        atr_period, atr_ratio, metrics = future.result()
                        if metrics:
                            results.append((atr_period, atr_ratio, metrics))
                            pbar.update(1)
                    except Exception as e:
                        print(f"Error in future: {e}")
                        continue

        print(f"Number of results collected: {len(results)}")
        if not results:
            print("No results to save. Exiting...")
            return

        os.makedirs("backtest_results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results/{args.symbol}_{args.timeframe}_{args.date_start}_{args.date_end}_{args.atr_period_range}_{args.atr_ratio_range}_{timestamp}.json"
        print(f"Attempting to save results to: {filename}")
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, default=str)
            logger.info(f"Backtest results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save backtest results to {filename}: {e}")
            raise

        if args.verbose and not stop_flag and results:
            file_handler = logging.FileHandler(f"logs/{current_time}_{args.timeframe}_{args.symbol}.log")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print_range_summary(results, args.symbol, args.timeframe, args.date_start, args.date_end,
                                args.starting_capital, args.top_n)

        if not stop_flag:
            print_range_summary(results, args.symbol, args.timeframe, args.date_start, args.date_end,
                                args.starting_capital, args.top_n)
        else:
            print("Backtest interrupted. Partial results may be available.")

        if file_handler:
            logger.removeHandler(file_handler)

    else:
        logger.info(f"Running backtest with ATR Period={args.atr_period}, ATR Ratio={args.atr_ratio}")
        trades, total_profit, df_with_indicators = backtest_strategy(df, args.atr_period, args.atr_ratio, start_date, end_date)
        if not stop_flag:
            print_trade_summary(trades, total_profit, args.symbol, args.timeframe, args.date_start, args.date_end,
                                args.atr_period, args.atr_ratio, args.starting_capital)
            if args.chart:
                print("Generating Supertrend chart...")
                output_filename = f"supertrend_chart_{args.symbol}_{args.timeframe}_{args.date_start}_{args.date_end}.html"
                plot_supertrend_chart(df_with_indicators, trades, args.date_start, args.date_end, args.symbol,
                                      args.timeframe, output_filename=output_filename)
        else:
            print("Backtest interrupted.")

if __name__ == '__main__':
    main()