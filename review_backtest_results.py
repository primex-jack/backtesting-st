import os
import json
import pandas as pd
import argparse
from datetime import datetime

def load_results(folder):
    """Load all backtest result files from the specified folder."""
    results = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                for atr_period, atr_ratio, metrics in data:
                    if metrics:  # Ensure metrics is not None
                        metrics['atr_period'] = atr_period
                        metrics['atr_ratio'] = atr_ratio
                        metrics['filename'] = filename  # Add filename for reference
                        results.append(metrics)
    return results

def print_results_table(df):
    """Print the DataFrame as a formatted table."""
    if df.empty:
        print("No results to display.")
        return

    # Define the table header
    header = (
        "| ATR_PERIOD | ATR_RATIO | Trades | # Wins | # Losses | % Profitable | Total P/L  | Open P/L  | Biggest Profit  | Biggest Loss  | Max Drawdown  | Max Runup  |"
    )
    separator = (
        "|------------|-----------|--------|--------|----------|--------------|------------|-----------|-----------------|---------------|---------------|------------|"
    )

    print("=" * 80)
    print("Filtered Backtest Results")
    print("=" * 80)
    print(header)
    print(separator)

    # Iterate over the DataFrame rows and print each row as a table entry
    for _, row in df.iterrows():
        row_str = (
            f"| {row['atr_period']:<10} | {row['atr_ratio']:<9.1f} | {row['num_trades']:>6} | "
            f"{row['winning_trades']:>6} | {row['losing_trades']:>8} | "
            f"{row['percent_profitable']:>11.2f}% | ${row['total_profit']:>9.2f} | "
            f"${row['unrealized_pl']:>8.2f} | ${row['largest_winning_trade']:>14.2f} | "
            f"${row['largest_losing_trade']:>12.2f} | ${row['max_drawdown']:>12.2f} | "
            f"${row['max_runup']:>9.2f} |"
        )
        print(row_str)

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Review Supertrend Backtest Results')
    parser.add_argument('--folder', default='backtest_results', help='Folder containing result files')
    parser.add_argument('--symbol', help='Filter by trading pair (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', help='Filter by timeframe (e.g., 15m)')
    parser.add_argument('--date-start', help='Filter by start date (YYYY-MM-DD)')
    parser.add_argument('--date-end', help='Filter by end date (YYYY-MM-DD)')
    parser.add_argument('--atr-period-min', type=int, help='Minimum ATR period')
    parser.add_argument('--atr-period-max', type=int, help='Maximum ATR period')
    parser.add_argument('--atr-ratio-min', type=float, help='Minimum ATR ratio')
    parser.add_argument('--atr-ratio-max', type=float, help='Maximum ATR ratio')
    parser.add_argument('--min-profit', type=float, help='Minimum total profit')
    parser.add_argument('--max-drawdown', type=float, help='Maximum drawdown threshold')
    parser.add_argument('--min-runup', type=float, help='Minimum max runup threshold')
    parser.add_argument('--sort-by', default='total_profit', choices=['total_profit', 'max_drawdown', 'max_runup', 'percent_profitable', 'atr_period', 'atr_ratio'], help='Sort by field')
    parser.add_argument('--ascending', action='store_true', help='Sort in ascending order')
    parser.add_argument('--descending', action='store_true', help='Sort in descending order')
    parser.add_argument('--output-file', help='Save filtered results to file (JSON or CSV)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of results (0 for all)')

    args = parser.parse_args()

    # Load all results
    results = load_results(args.folder)
    if not results:
        print("No backtest results found in the specified folder.")
        return

    # Debug: Print number of results loaded
    print(f"Loaded {len(results)} results from {args.folder}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Debug: Print total_profit range before filtering
    if not df.empty:
        print(f"Total Profit Range - Min: {df['total_profit'].min()}, Max: {df['total_profit'].max()}")

    # Extract date range from filename for filtering
    df['date_start'] = df['filename'].apply(lambda x: x.split('_')[2])
    df['date_end'] = df['filename'].apply(lambda x: x.split('_')[3])

    # Apply filters
    if args.symbol:
        df = df[df['filename'].str.contains(args.symbol, case=False)]
        print(f"After symbol filter ({args.symbol}): {len(df)} results")
    if args.timeframe:
        df = df[df['filename'].str.contains(args.timeframe, case=False)]
        print(f"After timeframe filter ({args.timeframe}): {len(df)} results")
    if args.date_start:
        df = df[pd.to_datetime(df['date_start']) >= pd.to_datetime(args.date_start)]
        print(f"After date_start filter ({args.date_start}): {len(df)} results")
    if args.date_end:
        df = df[pd.to_datetime(df['date_end']) <= pd.to_datetime(args.date_end)]
        print(f"After date_end filter ({args.date_end}): {len(df)} results")
    if args.atr_period_min:
        df = df[df['atr_period'] >= args.atr_period_min]
        print(f"After atr_period_min filter ({args.atr_period_min}): {len(df)} results")
    if args.atr_period_max:
        df = df[df['atr_period'] <= args.atr_period_max]
        print(f"After atr_period_max filter ({args.atr_period_max}): {len(df)} results")
    if args.atr_ratio_min:
        df = df[df['atr_ratio'] >= args.atr_ratio_min]
        print(f"After atr_ratio_min filter ({args.atr_ratio_min}): {len(df)} results")
    if args.atr_ratio_max:
        df = df[df['atr_ratio'] <= args.atr_ratio_max]
        print(f"After atr_ratio_max filter ({args.atr_ratio_max}): {len(df)} results")
    if args.min_profit:
        df = df[df['total_profit'] >= args.min_profit]
        print(f"After min_profit filter ({args.min_profit}): {len(df)} results")
    if args.max_drawdown:
        df = df[df['max_drawdown'] <= args.max_drawdown]
        print(f"After max_drawdown filter ({args.max_drawdown}): {len(df)} results")
    if args.min_runup:
        df = df[df['max_runup'] >= args.min_runup]
        print(f"After min_runup filter ({args.min_runup}): {len(df)} results")

    if df.empty:
        print("No results match the specified filters.")
        return

    # Sort
    ascending = args.ascending if args.ascending else (not args.descending)
    df = df.sort_values(by=args.sort_by, ascending=ascending)

    # Limit results
    if args.limit > 0:
        df = df.head(args.limit)

    # Display results as a formatted table
    print_results_table(df)

    # Save to file if specified
    if args.output_file:
        if args.output_file.endswith('.json'):
            df.to_json(args.output_file, orient='records')
        elif args.output_file.endswith('.csv'):
            df.to_csv(args.output_file, index=False)
        else:
            print("Unsupported file format. Use .json or .csv.")
            return
        print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()