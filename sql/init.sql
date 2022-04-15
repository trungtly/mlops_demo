-- Initialize fraud detection database
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE,
    features JSONB,
    prediction_score FLOAT,
    prediction_label INTEGER,
    model_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feedback_label INTEGER DEFAULT NULL,
    feedback_at TIMESTAMP DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    dataset_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    drift_score FLOAT,
    drift_detected BOOLEAN,
    report_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_model_metrics_created_at ON model_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_reports_created_at ON drift_reports(created_at);