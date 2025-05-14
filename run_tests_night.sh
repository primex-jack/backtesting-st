#!/bin/bash
echo "Starting backtests for multiple timeframes at $(date)..."
echo "----------------------------------------------------------------"
# Function to run a backtest
run_backtest() {
    local symbol=$1
    local tf=$2
    local log_file="backtest_${symbol}_${tf}_2024__09-2025_04-05.log"
    echo "Running backtest for ${symbol}/USDT timeframe $tf at $(date)..."
    python supertrend_backtester.py --symbol "${symbol}USDT" --timeframe "$tf" \
        --date-start 2024-07-02 --date-end 2025-04-05 \
        --atr-period-range 2-98 --atr-ratio-range 2-98 \
        --top-n 300 --workers 32 --quiet > "$log_file" 2>&1
    echo "Backtest for ${symbol}/USDT timeframe $tf completed at $(date). Logged to $log_file"
}

echo "----------------------------------------------------------------"
echo "--------------------------  BNB  ------------------------------"
# BNB/USDT backtests
for tf in "4h" "2h" "1h" "30m" "15m" "5m" "3m"; do
    run_backtest "BNB" "$tf"
done


echo "----------------------------------------------------------------"
echo "--------------------------  ETH  ------------------------------"
# ETH/USDT backtests
for tf in "4h" "2h" "1h" "30m" "15m" "5m" "3m"; do
    run_backtest "ETH" "$tf"
done

echo "----------------------------------------------------------------"
echo "--------------------------  ARB  ------------------------------"
# ARB/USDT backtests
for tf in "4h" "2h" "1h" "30m" "15m" "5m" "3m"; do
    run_backtest "ARB" "$tf"
done

#echo "----------------------------------------------------------------"
#echo "--------------------------  AVAX  ------------------------------"
# AVAX/USDT backtests
#for tf in "1m"; do
#    run_backtest "AVAX" "$tf"
#done


#echo "----------------------------------------------------------------"
#echo "--------------------------   ADA  ------------------------------"
# ADA/USDT backtests
#for tf in "1m"; do
#    run_backtest "ADA" "$tf"
#done

echo "All backtests completed at $(date)!"
