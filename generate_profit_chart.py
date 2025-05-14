import os
import glob
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import argparse
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
HOME_DIR = Path.home()
BASE_DIR = HOME_DIR / "backtest_project"
REPORTS_DIR = BASE_DIR / "account_value_reports"
OUTPUT_CSV_CUMULATIVE = BASE_DIR / "cumulative_profits.csv"
OUTPUT_CSV_TRADES = BASE_DIR / "trade_details.csv"
OUTPUT_HTML = BASE_DIR / "account_value_chart.html"
OUTPUT_PNG = BASE_DIR / "account_value_chart.png"

# Ensure output directory exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate account value chart from backtest CSVs")
    parser.add_argument('--start-date', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--resample', type=str, default='1h', help="Resample timeframe (e.g., 1h, 1d, None to disable)")
    return parser.parse_args()

def resample_dataframe(df, timeframe):
    """Resample the DataFrame to the specified timeframe, taking the last value in each window."""
    if timeframe.lower() == 'none':
        return df
    
    logger.info(f"Resampling DataFrame to {timeframe} timeframe")
    # Resample to the specified timeframe, taking the last value for each column
    df_resampled = df.resample(timeframe).last()
    # Forward-fill any missing values (though there shouldn't be many after taking the last value)
    df_resampled = df_resampled.ffill()
    logger.info(f"Resampled DataFrame from {len(df)} to {len(df_resampled)} records")
    return df_resampled

def main():
    args = parse_args()
    # Find all CSV files
    csv_files = glob.glob(str(REPORTS_DIR / "*.csv"))
    if not csv_files:
        print("No CSV files found in", REPORTS_DIR)
        return
    
    all_trades = []
    report_data = {}
    account_value_data = {}
    
    # Process each CSV file
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            report_label = os.path.basename(filepath).replace('.csv', '')
            required_cols = ['timestamp', 'trading_pair', 'timeframe', 'trade_type', 'entry_price', 'exit_price', 'size', 'profit_per_contract', 'normalized_profit', 'cumulative_profit', 'floating_pl', 'account_value']
            if not all(col in df for col in required_cols):
                print(f"Warning: Missing required columns in {filepath}: {set(required_cols) - set(df.columns)}")
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Check for duplicate timestamps
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                print(f"Warning: Found {duplicates} duplicate timestamps in {filepath}. Keeping last row per timestamp.")
                df = df.drop_duplicates(subset='timestamp', keep='last')
            # Filter by date range
            if args.start_date:
                df = df[df['timestamp'] >= pd.to_datetime(args.start_date)]
            if args.end_date:
                df = df[df['timestamp'] <= pd.to_datetime(args.end_date)]
            if df.empty:
                print(f"Warning: No data after date filtering in {filepath}")
                continue
            # Store trade details
            trade_df = df[df['trade_type'] != 'None'][required_cols]
            trade_df['report'] = report_label
            all_trades.append(trade_df)
            # Resample the DataFrame to the specified timeframe
            df = df.set_index('timestamp')
            df = resample_dataframe(df, args.resample)
            # Store full DataFrame and account value separately
            report_data[report_label] = df
            account_value_data[report_label] = df[['account_value']].rename(columns={'account_value': report_label})
            logger.info(f"Parsed {filepath}: {len(df)} records, columns in report_data: {list(report_data[report_label].columns)}")
            print(f"Parsed {filepath}: {len(df)} records")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not report_data:
        print("No valid data to process")
        return
    
    # Save trade details to CSV
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trades_df.empty:
        trades_df.sort_values(['timestamp', 'report']).to_csv(OUTPUT_CSV_TRADES, index=False)
        print(f"Saved trade details to {OUTPUT_CSV_TRADES}")
    
    # Combine account values
    combined_df = pd.concat([df for df in account_value_data.values()], axis=1)
    combined_df = combined_df.sort_index().ffill()
    # Add total account value
    combined_df['Total'] = combined_df[[col for col in combined_df.columns if col != 'Total']].sum(axis=1)
    
    # Save cumulative profits to CSV
    output_df = combined_df.reset_index().rename(columns={'timestamp': 'timestamp'})
    output_df.to_csv(OUTPUT_CSV_CUMULATIVE, index=False)
    print(f"Saved account values to {OUTPUT_CSV_CUMULATIVE}")
    
    # Create Plotly chart
    fig = go.Figure()
    for label in combined_df.columns:
        report_df = report_data.get(label, pd.DataFrame())
        logger.info(f"Processing label {label}, columns available: {list(report_df.columns)}")
        if label == 'Total':
            # For Total, use zero-filled customdata and simpler hover template
            customdata = np.zeros((len(combined_df), 2))  # Zero-filled for cumulative_profit, floating_pl
            hover_template = "Report: %{text}<br>Date: %{x}<br>Account Value: $%{y:,.2f}"
        else:
            # For individual reports, use cumulative_profit and floating_pl
            customdata = report_df.reindex(combined_df.index)[['cumulative_profit', 'floating_pl']].fillna(0).values
            hover_template = "Report: %{text}<br>Date: %{x}<br>Account Value: $%{y:,.2f}<br>Realized P/L: $%{customdata[0]:,.2f}<br>Floating P/L: $%{customdata[1]:,.2f}"
        
        line_style = dict(shape='hv')
        if label == 'Total':
            line_style.update(dash='dash', width=3)
        
        fig.add_trace(go.Scatter(
            x=combined_df.index,
            y=combined_df[label],
            mode='lines',
            name=label,
            line=line_style,
            text=[label] * len(combined_df),
            customdata=customdata,
            hovertemplate=hover_template
        ))
    
    # Configure layout
    fig.update_layout(
        title="Account Value Across Backtests",
        xaxis_title="Date",
        yaxis_title="Account Value (USD)",
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )
    
    # Save HTML
    fig.write_html(str(OUTPUT_HTML))
    print(f"Saved interactive chart to {OUTPUT_HTML}")
    
    # Save PNG
    try:
        pio.write_image(fig, file=str(OUTPUT_PNG), format='png')
        print(f"Saved static chart to {OUTPUT_PNG}")
    except Exception as e:
        print(f"Failed to save PNG: {e}")
    
    print(f"Processed {len(csv_files)} CSV files")

if __name__ == "__main__":
    main()
