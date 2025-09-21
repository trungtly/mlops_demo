# Model Monitoring Guide

This guide explains how to use the fraud detection monitoring tools to detect data drift and model performance degradation.

## Overview

The monitoring system consists of two main components:

1. **Data Drift Detection**: Identifies when the statistical properties of your input data change significantly from the baseline.
2. **Performance Monitoring**: Tracks model performance metrics over time and alerts when they degrade.

## Quick Start

To quickly test the monitoring system:

```bash
# Step 1: Train a simple model and create test datasets
python scripts/quick_train_model.py

# Step 2: Run the monitoring script
python scripts/run_monitoring.py \
    --config-path configs/monitoring.yaml \
    --reference-data-path data/reference/reference_data.csv \
    --current-data-path data/current/current_data.csv \
    --model-path models/fraud_detection_model.pkl \
    --output-dir monitoring_reports/demo
```

## Configuration

The monitoring system is configured via `configs/monitoring.yaml`. Key settings include:

```yaml
# Features to monitor for drift
features_to_monitor:
  - V1
  - V2
  # ... add more features

# Baseline metrics for model performance comparison
baseline_metrics:
  accuracy: 0.999
  precision: 0.864
  recall: 0.901
  f1_score: 0.882
  roc_auc: 0.987
  pr_auc: 0.842

# Drift detection thresholds
drift_thresholds:
  ks_p_value: 0.05
  psi_minor: 0.1
  psi_major: 0.25

# Alert thresholds for performance degradation
performance_alert_thresholds:
  warning_degradation_pct: 5.0
  critical_degradation_pct: 10.0
```

## Understanding Drift Metrics

### Population Stability Index (PSI)

PSI measures the difference between distributions:

- **PSI < 0.1**: No significant drift
- **0.1 ≤ PSI < 0.25**: Minor drift detected
- **PSI ≥ 0.25**: Major drift detected

### Kolmogorov-Smirnov (KS) Test

The KS test is a statistical test that determines if two samples come from the same distribution:

- **p-value < 0.05**: Significant difference detected (drift)
- **p-value ≥ 0.05**: No significant difference

## Performance Monitoring

Performance metrics are compared against the baseline with thresholds:

- **Degradation < warning_threshold**: Status = OK
- **warning_threshold ≤ Degradation < critical_threshold**: Status = WARNING
- **Degradation ≥ critical_threshold**: Status = CRITICAL

## Reports and Visualizations

The monitoring system generates several outputs:

1. **Drift Report CSV**: Details about data drift for each feature
2. **Performance Report YAML/JSON**: Model performance metrics and comparison
3. **Summary JSON**: High-level overview with alerts
4. **Visualizations**:
   - Feature distribution comparisons
   - PSI bar chart
   - Performance metrics radar chart
   - Dataset comparison chart

## Integration with MLOps Workflow

For production use, integrate monitoring into your MLOps workflow:

1. **Regular Monitoring**: Schedule runs using cron or an orchestrator
2. **Alert Integration**: Connect alerts to notification systems
3. **Dashboard Integration**: Visualize trends over time
4. **Automated Retraining**: Trigger retraining when drift thresholds are exceeded

## Advanced Usage

### Custom Monitoring Scripts

For more advanced scenarios, you can create a custom monitoring script:

```python
from fraud_detection.monitoring.drift import DataDriftMonitor
from fraud_detection.monitoring.performance import PerformanceMonitor

# Initialize monitors
drift_monitor = DataDriftMonitor(reference_df, current_df)
perf_monitor = PerformanceMonitor(model_path, y_true, y_pred, y_prob)

# Get drift report
drift_report = drift_monitor.get_drift_report()

# Get performance comparison
performance = perf_monitor.compare_with_baseline(baseline_metrics)
```

### Adding Custom Metrics

To add custom drift metrics or performance metrics, extend the corresponding classes in:

- `src/fraud_detection/monitoring/drift.py`
- `src/fraud_detection/monitoring/performance.py`

## Troubleshooting

Common issues:

1. **Missing reference data**: Ensure reference data is available at the specified path
2. **Model format mismatch**: Verify model is saved with the correct serialization format
3. **Feature mismatch**: Ensure reference and current data have the same features
4. **Configuration errors**: Validate the YAML configuration file format

## Best Practices

1. Use a representative sample of your training data as reference data
2. Monitor both data drift and model performance together
3. Set appropriate thresholds based on your specific use case
4. Review and adjust thresholds periodically
5. Maintain a versioned history of monitoring reports
6. Automate the monitoring workflow