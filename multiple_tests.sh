#!/bin/bash

# Fixed settings
DATE_START="2024-07-01"
DATE_END="2025-05-11"
WORKERS=1
OUTPUT_DIR="single_reports"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Settings file (required)
SETTINGS_FILE="settings.txt"

# Check if settings file exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Error: $SETTINGS_FILE not found. Please create it with the format: symbol,timeframe,atr-period,atr-ratio,starting-capital,trade-size"
    exit 1
fi

# Load settings from external file
echo "Loading settings from $SETTINGS_FILE"
mapfile -t SETTINGS < "$SETTINGS_FILE"

# Batch size limit
BATCH_SIZE=20
CURRENT_BATCH=0

# Function to run a single backtest
run_backtest() {
    local symbol="$1"
    local timeframe="$2"
    local atr_period="$3"
    local atr_ratio="$4"
    local starting_capital="$5"
    local trade_size="$6"
    
    # Construct the output log file name
    log_file="${OUTPUT_DIR}/${symbol}_${timeframe}_atr${atr_period}_ratio${atr_ratio}.log"
    
    # Run the command and redirect output to log file
    echo "Starting backtest for $symbol, $timeframe, ATR: $atr_period, Ratio: $atr_ratio, Starting Capital: $starting_capital, Trade Size: $trade_size"
    python3 charted_supertrend_backtester.py \
        --symbol "$symbol" \
        --timeframe "$timeframe" \
        --date-start "$DATE_START" \
        --date-end "$DATE_END" \
        --atr-period "$atr_period" \
        --atr-ratio "$atr_ratio" \
        --starting-capital "$starting_capital" \
        --trade-size "$trade_size" \
        --workers "$WORKERS" \
        --quiet > "$log_file" 2>&1
}

# Export the function to be used by parallel
export -f run_backtest
export DATE_START DATE_END WORKERS OUTPUT_DIR

# Loop through each setting combination and run in batches
for setting in "${SETTINGS[@]}"; do
    # Split the setting string into variables
    IFS=',' read -r symbol timeframe atr_period atr_ratio starting_capital trade_size <<< "$setting"
    
    # Run backtest in background
    run_backtest "$symbol" "$timeframe" "$atr_period" "$atr_ratio" "$starting_capital" "$trade_size" &
    
    # Increment batch counter
    ((CURRENT_BATCH++))
    
    # If batch size is reached, wait for current batch to complete
    if [ "$CURRENT_BATCH" -ge "$BATCH_SIZE" ]; then
        wait
        CURRENT_BATCH=0
        echo "Completed batch of $BATCH_SIZE tests. Starting next batch..."
    fi
done

# Wait for any remaining background processes to complete
wait

echo "All backtests completed. Results saved in $OUTPUT_DIR"
