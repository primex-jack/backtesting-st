#!/bin/bash
echo "Starting backtests for multiple timeframes..."

for tf in "30m" "1h"; do
    echo "Running backtest for SOL/USDT timeframe $tf..."
    python supertrend_backtester.py --symbol SOLUSDT --timeframe "$tf" --date-start 2025-01-02 --date-end 2025-03-11 --atr-period-range 2-99 --atr-ratio-range 2-99 --top-n 300 --workers 32 --quiet > "backtest_sol_${tf}.log" 2>&1
    echo "Backtest for timeframe $tf completed. Logged to backtest_sol_${tf}.log"
done

for tf in "5m" "15m" "30m"; do
    echo "Running backtest for BNB/USDT timeframe $tf..."
    python supertrend_backtester.py --symbol BNBUSDT --timeframe "$tf" --date-start 2025-01-02 --date-end 2025-03-11 --atr-period-range 2-99 --atr-ratio-range 2-99 --top-n 300 --workers 32 --quiet > "backtest_bnb_${tf}.log" 2>&1
    echo "Backtest for timeframe $tf completed. backtest_bnb_${tf}.log"
done

echo "All backtests completed!"

