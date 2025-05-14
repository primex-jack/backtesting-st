import sys
import pandas as pd
from binance.client import Client
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime


def fetch_klines(client, symbol, interval, start_ts, end_ts, use_futures=True):
    klines = []
    limit = 100  # API request limit per batch
    method = client.futures_historical_klines if use_futures else client.get_historical_klines

    interval_ms = interval_to_milliseconds(interval)
    if interval_ms is None:
        print(
            f"Unsupported interval: {interval}. Supported intervals are: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.")
        return None

    total_steps = (end_ts - start_ts) // (limit * interval_ms)

    with tqdm(total=total_steps,
              desc=f"Fetching klines for {symbol} {'(futures)' if use_futures else '(spot)'})") as pbar:
        current_start_ts = start_ts
        while current_start_ts < end_ts:
            batch = method(
                symbol=symbol,
                interval=interval,
                start_str=str(current_start_ts),
                end_str=str(end_ts),
                limit=limit
            )
            if not batch:
                break
            klines.extend(batch)
            if batch:
                current_start_ts = int(batch[-1][0]) + interval_ms
            else:
                break
            pbar.update(1)
    return klines


def interval_to_milliseconds(interval):
    """Convert Binance interval string to milliseconds."""
    interval_map = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
        '1M': 30 * 24 * 60 * 60 * 1000
    }
    return interval_map.get(interval, None)


def validate_data(df, interval, start_str, end_str):
    if df is None or df.empty:
        return False, "DataFrame is empty or None."

    interval_ms = interval_to_milliseconds(interval)
    if interval_ms is None:
        return False, f"Unsupported interval: {interval}"

    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)

    first_ts = int(df.index[0].timestamp() * 1000)
    last_ts = int(df.index[-1].timestamp() * 1000)

    print(f"Validating data: Data range from {df.index[0]} ({first_ts}) to {df.index[-1]} ({last_ts})")
    print(f"Requested range: {start_str} ({start_ts}) to {end_str} ({end_ts})")

    if first_ts > start_ts + interval_ms:
        return False, f"Data does not cover the start period: starts at {df.index[0]}, needed {start_str}"

    # Allow a tolerance of two intervals for the end time
    if last_ts < end_ts - 2 * interval_ms:
        return False, f"Data does not cover the end period: ends at {df.index[-1]}, needed {end_str}"

    time_diffs = df.index.to_series().diff().dt.total_seconds() * 1000
    time_diffs = time_diffs.iloc[1:]

    expected_diff_ms = interval_ms
    max_allowed_diff = expected_diff_ms * 1.5

    missing_intervals = time_diffs[time_diffs > max_allowed_diff]
    if not missing_intervals.empty:
        print(f"Missing intervals at: {missing_intervals.index}")
        return False, f"Found {len(missing_intervals)} missing intervals in the data."

    if df.index.duplicated().any():
        return False, f"Found duplicate timestamps in the data: {df.index[df.index.duplicated()]}"

    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    if df[ohlcv_columns].isna().any().any():
        return False, "Found NaN values in OHLCV data."

    # Separate validation for prices and volume
    price_columns = ['open', 'high', 'low', 'close']
    volume_column = ['volume']

    # Check for non-positive prices and log the offending rows
    non_positive_prices = (df[price_columns] <= 0).any(axis=1)
    if non_positive_prices.any():
        print("Found non-positive price values in OHLC data at the following timestamps:")
        print(df[non_positive_prices][price_columns])
        df.drop(index=df[non_positive_prices].index, inplace=True)
        if df.empty:
            return False, "DataFrame is empty after removing non-positive price values."
        print(f"Dropped {non_positive_prices.sum()} rows with non-positive price values.")

    # Replace non-positive volume with 1
    non_positive_volume = (df[volume_column] <= 0).any(axis=1)
    if non_positive_volume.any():
        print("Found non-positive volume values at the following timestamps:")
        print(df[non_positive_volume][volume_column])
        df.loc[non_positive_volume, 'volume'] = 1.0
        print(f"Replaced {non_positive_volume.sum()} non-positive volume values with 1.0.")

    return True, "Data validation passed successfully."


def process_klines(klines, interval):
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['timestamp'] = df['timestamp'].astype(int)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def load_existing_data(filename, symbol, interval, start_str, end_str):
    if not os.path.exists(filename):
        return None

    metadata_file = f"{filename}.meta"
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if metadata['symbol'] != symbol or metadata['interval'] != interval:
            return None

        df_existing = pd.read_csv(filename, index_col='datetime', parse_dates=True)
        if df_existing.empty:
            return None

        existing_start = metadata['start_str']
        existing_end = metadata['end_str']
        requested_start = pd.to_datetime(start_str)
        requested_end = pd.to_datetime(end_str)
        existing_start_ts = int(pd.to_datetime(existing_start).timestamp() * 1000)
        existing_end_ts = int(pd.to_datetime(existing_end).timestamp() * 1000)
        requested_start_ts = int(requested_start.timestamp() * 1000)
        requested_end_ts = int(requested_end.timestamp() * 1000)

        print(f"Existing data range: {existing_start} ({existing_start_ts}) to {existing_end} ({existing_end_ts})")
        print(f"Requested data range: {start_str} ({requested_start_ts}) to {end_str} ({requested_end_ts})")

        if (existing_start_ts <= requested_start_ts and
                existing_end_ts >= requested_end_ts):
            is_valid, validation_message = validate_data(df_existing, interval, start_str, end_str)
            if is_valid:
                return df_existing
            else:
                print(f"Existing data validation failed: {validation_message}")
                return None

        return df_existing, existing_start_ts, existing_end_ts
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return None


def save_data_with_metadata(df, filename, symbol, interval, start_str, end_str):
    # Update end_str to reflect the actual last timestamp
    actual_end_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(filename)
    metadata = {
        'symbol': symbol,
        'interval': interval,
        'start_str': start_str,
        'end_str': actual_end_str,
        'saved_at': datetime.now().isoformat()
    }
    metadata_file = f"{filename}.meta"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def load_or_update_klines(client, symbol, interval, start_str, end_str):
    try:
        KLINES_FILE = f'{symbol.lower()}_{interval}_klines.csv'
        interval_ms = interval_to_milliseconds(interval)
        if interval_ms is None:
            print(
                f"Unsupported interval: {interval}. Supported intervals are: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.")
            return None

        existing_data = load_existing_data(KLINES_FILE, symbol, interval, start_str, end_str)
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)

        if existing_data is not None:
            if isinstance(existing_data, tuple):
                print(f"Loaded partial data from {KLINES_FILE}, fetching additional data...")
                df_existing, existing_start_ts, existing_end_ts = existing_data
                dataframes = []
                if existing_start_ts > start_ts:
                    klines = fetch_klines(client, symbol, interval, start_ts, existing_start_ts - interval_ms,
                                          use_futures=True)
                    if klines:
                        df_new = process_klines(klines, interval)
                        if not df_new.empty:
                            dataframes.append(df_new)
                            print(f"Fetched earlier data: {df_new.index[0]} to {df_new.index[-1]}")
                    else:
                        klines = fetch_klines(client, symbol, interval, start_ts, existing_start_ts - interval_ms,
                                              use_futures=False)
                        if klines:
                            df_new = process_klines(klines, interval)
                            if not df_new.empty:
                                dataframes.append(df_new)
                                print(f"Fetched earlier data (spot): {df_new.index[0]} to {df_new.index[-1]}")
                if df_existing is not None and not df_existing.empty:
                    dataframes.append(df_existing)
                if existing_end_ts < end_ts:
                    klines = fetch_klines(client, symbol, interval, existing_end_ts + interval_ms, end_ts,
                                          use_futures=True)
                    if klines:
                        df_new = process_klines(klines, interval)
                        if not df_new.empty:
                            dataframes.append(df_new)
                            print(f"Fetched later data: {df_new.index[0]} to {df_new.index[-1]}")
                    else:
                        klines = fetch_klines(client, symbol, interval, existing_end_ts + interval_ms, end_ts,
                                              use_futures=False)
                        if klines:
                            df_new = process_klines(klines, interval)
                            if not df_new.empty:
                                dataframes.append(df_new)
                                print(f"Fetched later data (spot): {df_new.index[0]} to {df_new.index[-1]}")
                if not dataframes:
                    print("No dataframes to concatenate after fetching additional data.")
                    return None
                df = pd.concat(dataframes)
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)

                # Validate after concatenation but before saving
                is_valid, validation_message = validate_data(df, interval, start_str, end_str)
                if not is_valid:
                    print(f"Data validation failed after concatenation: {validation_message}")
                    return None

                save_data_with_metadata(df, KLINES_FILE, symbol, interval, start_str, end_str)
                print(f"Updated data saved to {KLINES_FILE}")
                return df
            else:
                print(f"Loaded data from {KLINES_FILE} (fully cached)")
                df = existing_data

                # Validate the fully cached data
                is_valid, validation_message = validate_data(df, interval, start_str, end_str)
                if not is_valid:
                    print(f"Data validation failed for fully cached data: {validation_message}")
                    return None

                return df
        else:
            print(f"No cached data found or validation failed, fetching data for {symbol}...")
            klines = fetch_klines(client, symbol, interval, start_ts, end_ts, use_futures=True)
            if klines is None or not klines:
                klines = fetch_klines(client, symbol, interval, start_ts, end_ts, use_futures=False)

            if klines is None or not klines:
                print(f"No klines retrieved for {symbol}. Exiting.")
                return None

            df = process_klines(klines, interval)

            is_valid, validation_message = validate_data(df, interval, start_str, end_str)
            if not is_valid:
                print(f"Data validation failed for newly fetched data: {validation_message}")
                return None

            save_data_with_metadata(df, KLINES_FILE, symbol, interval, start_str, end_str)
            print(f"New data saved to {KLINES_FILE}")
            return df
    except Exception as e:
        print(f"Error in load_or_update_klines: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Binance Historical Data Loader')
    parser.add_argument('--symbol', default='ETHUSDT', help='Trading pair (e.g., ETHUSDT)')
    parser.add_argument('--timeframe', default='3m',
                        help='Timeframe (e.g., 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)')
    parser.add_argument('--date-start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date-end', default='2023-01-02', help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    client = Client()

    print(f"Loading data for {args.symbol}...")
    data = load_or_update_klines(
        client, args.symbol, args.timeframe, args.date_start, args.date_end
    )
    if data is None:
        print("Failed to load data or validation failed. Exiting.")
        sys.exit(1)

    print(
        f"Data after filtering: {len(data)} bars from {data.index[0] if not data.empty else 'N/A'} to {data.index[-1] if not data.empty else 'N/A'}")
    print(f"Sample data:\n{data.head()}")


if __name__ == '__main__':
    main()