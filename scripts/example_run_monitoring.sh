#!/bin/bash
# Example script to run model monitoring

# Create directories if needed
mkdir -p data/reference
mkdir -p data/current
mkdir -p monitoring_reports

# Check if model exists
if [ ! -f "models/final_fraud_detection_model.pkl" ]; then
    echo "Error: Model file not found. Please run training first."
    exit 1
fi

# Define paths
REFERENCE_DATA="data/reference/reference_data.csv"
CURRENT_DATA="data/current/current_data.csv"
MODEL_PATH="models/final_fraud_detection_model.pkl"
CONFIG_PATH="configs/monitoring.yaml"
OUTPUT_DIR="monitoring_reports/$(date +%Y%m%d)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running model monitoring..."
echo "Reference data: $REFERENCE_DATA"
echo "Current data: $CURRENT_DATA"
echo "Model path: $MODEL_PATH"
echo "Config path: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run monitoring
python scripts/run_monitoring.py \
    --config-path "$CONFIG_PATH" \
    --reference-data-path "$REFERENCE_DATA" \
    --current-data-path "$CURRENT_DATA" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR"

# Check if monitoring was successful
if [ $? -eq 0 ]; then
    echo "Monitoring completed successfully."
    echo "Reports saved to $OUTPUT_DIR"
    
    # Display summary
    SUMMARY_FILE=$(ls -t "$OUTPUT_DIR"/summary_*.json | head -1)
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary:"
        cat "$SUMMARY_FILE" | python -m json.tool
    fi
else
    echo "Error running monitoring."
    exit 1
fi

echo "Done."