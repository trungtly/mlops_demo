#!/bin/bash
# Script to train a model and run monitoring

# Create and activate virtual environment if it doesn't exist
if [ ! -d "monitoring_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv monitoring_venv
    source monitoring_venv/bin/activate
    pip install pandas numpy scikit-learn joblib matplotlib seaborn pyyaml click
else
    source monitoring_venv/bin/activate
fi

# Check if model exists, if not train it
if [ ! -f "models/fraud_detection_model.pkl" ]; then
    echo "Training model..."
    python3 scripts/quick_train_model.py
fi

# Create output directory
OUTPUT_DIR="monitoring_reports/$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

# Run monitoring
echo "Running monitoring..."
python3 scripts/run_monitoring.py \
    --config-path configs/monitoring.yaml \
    --reference-data-path data/reference/reference_data.csv \
    --current-data-path data/current/current_data.csv \
    --model-path models/fraud_detection_model.pkl \
    --output-dir "$OUTPUT_DIR"

# Check if monitoring was successful
if [ $? -eq 0 ]; then
    echo "Monitoring completed successfully."
    echo "Reports saved to $OUTPUT_DIR"
    
    # Find summary file
    SUMMARY_FILE=$(ls -t "$OUTPUT_DIR"/summary_*.json | head -1)
    if [ -n "$SUMMARY_FILE" ]; then
        echo "Summary:"
        cat "$SUMMARY_FILE" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))"
    fi
else
    echo "Error running monitoring."
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "Done."