#!/usr/bin/env python3
"""
Quick training script to create a model for monitoring demo.
Trains a simple model on the creditcard.csv dataset.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

def main():
    print("Quick model training for monitoring demo")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the creditcard dataset
    print("Loading credit card fraud dataset...")
    data_path = Path('data/raw/creditcard.csv')
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Using sample data instead...")
        data_path = Path('data/sample/creditcard_sample.csv')
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Fraud rate in training set: {y_train.mean():.4f}")
    print(f"Fraud rate in test set: {y_test.mean():.4f}")
    
    # Train a simple Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    print("Model trained successfully")
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Save the model
    model_path = "models/fraud_detection_model.pkl"
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    # Save a reference dataset for monitoring
    reference_data_path = "data/reference/reference_data.csv"
    os.makedirs("data/reference", exist_ok=True)
    
    print(f"Saving reference data to {reference_data_path}")
    
    # Take a sample of the test set as reference data
    reference_data = pd.DataFrame(X_test.copy())
    reference_data['Class'] = y_test.values
    reference_data.to_csv(reference_data_path, index=False)
    
    # Create a current data sample with some drift for testing
    current_data_path = "data/current/current_data.csv"
    os.makedirs("data/current", exist_ok=True)
    
    print(f"Creating current data with artificial drift to {current_data_path}")
    
    # Introduce some artificial drift
    current_data = reference_data.copy()
    
    # Shift V1, V2 slightly to introduce drift
    current_data['V1'] = current_data['V1'] * 1.1 + 0.1
    current_data['V2'] = current_data['V2'] * 0.9 - 0.2
    current_data['V3'] = current_data['V3'] + np.random.normal(0, 0.2, size=len(current_data))
    
    # Save the current data
    current_data.to_csv(current_data_path, index=False)
    
    print("Done! Now you can run the monitoring script:")
    print("python scripts/run_monitoring.py --config-path configs/monitoring.yaml " +
          f"--reference-data-path {reference_data_path} --current-data-path {current_data_path} " +
          f"--model-path {model_path} --output-dir monitoring_reports/demo")

if __name__ == "__main__":
    main()