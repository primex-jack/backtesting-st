import subprocess
import os
from datetime import datetime
import itertools
import colorama
from colorama import Fore, Style
import signal
import sys
from tqdm import tqdm

# Initialize colorama for Windows PowerShell compatibility
colorama.init()

# Global flag to handle interruption
stop_event = False

def signal_handler(sig, frame):
    global stop_event
    print(f"\n{Fore.RED}Received Ctrl+C, interrupting backtests...{Style.RESET_ALL}")
    stop_event = True

def run_backtest(symbol, timeframe, atr_period_range, atr_ratio_range, starting_capital, workers, top_n, output_file):
    global stop_event
    if stop_event:
        return

    # Construct the command to run supertrend_backtester.py
    command = [
        "python", "supertrend_backtester.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--date-start", "2024-10-01",
        "--date-end", "2025-03-12",
        "--atr-period-range", atr_period_range,
        "--atr-ratio-range", atr_ratio_range,
        "--starting-capital", str(starting_capital),
        "--workers", str(workers),
        "--top-n", str(top_n)
    ]

    # Print start message in green
    print(f"{Fore.GREEN}Starting backtest: {symbol} {timeframe} ATR {atr_period_range} Ratio {atr_ratio_range} at {datetime.now()}{Style.RESET_ALL}")

    # Run the command with real-time output to both console and file
    with open(output_file, 'a') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Backtest Run: {symbol} {timeframe} ATR {atr_period_range} Ratio {atr_ratio_range}\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write("=" * 100 + "\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        buffer = ""
        for line in process.stdout:
            if stop_event:
                break
            buffer += line
            # Check if the line contains a progress bar update (ends with '\r')
            if line.endswith('\r'):
                # Print the progress bar update using tqdm.write to preserve formatting
                tqdm.write(buffer.rstrip('\r'), end='\r')
                f.write(buffer)
                buffer = ""
            else:
                # Regular line, print and write as-is
                print(buffer, end='')
                f.write(buffer)
                buffer = ""
        # Flush any remaining buffer
        if buffer:
            print(buffer, end='')
            f.write(buffer)
        for line in process.stderr:
            print(line, end='')  # Print errors to console in real-time
            f.write(line)        # Write errors to file
        process.wait(timeout=3600)  # Optional timeout of 1 hour per backtest
        if process.returncode != 0:
            print(f"{Fore.RED}Error: Backtest failed with return code {process.returncode}{Style.RESET_ALL}")
            f.write(f"\nError: Backtest failed with return code {process.returncode}\n")
        elif stop_event:
            process.terminate()  # Terminate the process if interrupted
            print(f"{Fore.RED}Backtest terminated due to Ctrl+C{Style.RESET_ALL}")
            f.write(f"\nBacktest terminated due to Ctrl+C\n")

    # Print finish message in blue
    if not stop_event:
        print(f"{Fore.BLUE}Finished backtest: {symbol} {timeframe} ATR {atr_period_range} Ratio {atr_ratio_range} at {datetime.now()}{Style.RESET_ALL}")

def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Define parameters
    symbol = "BTCUSDT"
    timeframes = ["3m", "5m", "10m", "15m"]
    atr_period_ranges = [f"{start}-{end}" for start, end in [(5, 40)]]
    atr_ratio_ranges = [f"{start}-{end}" for start, end in [(20.0, 80.0)]]
    starting_capital = 100000
    workers = 8
    top_n = 50  # Display top 50 results

    # Define output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"logs/backtest_results_{timestamp}.txt"

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Write initial header to the file
    with open(output_file, 'w') as f:
        f.write("Backtest Results\n")
        f.write("=" * 100 + "\n\n")

    # Iterate over all combinations
    for timeframe in timeframes:
        for atr_period_range, atr_ratio_range in itertools.product(atr_period_ranges, atr_ratio_ranges):
            run_backtest(symbol, timeframe, atr_period_range, atr_ratio_range, starting_capital, workers, top_n, output_file)
            if stop_event:
                break
        if stop_event:
            break

    # Print completion message in yellow if not interrupted
    if not stop_event:
        print(f"{Fore.YELLOW}All backtests completed. Results saved to: {output_file}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Backtests interrupted. Partial results saved to: {output_file}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()