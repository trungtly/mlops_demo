import click
import pandas as pd
import joblib
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_monitoring')

class DataDriftMonitor:
    """Class to handle data drift detection."""
    
    def __init__(self, reference_df, current_df, config=None):
        """Initialize with reference and current datasets."""
        self.reference_df = reference_df
        self.current_df = current_df
        self.config = config or {}
        
        # Set thresholds from config or use defaults
        self.ks_threshold = self.config.get('drift_thresholds', {}).get('ks_p_value', 0.05)
        self.psi_minor_threshold = self.config.get('drift_thresholds', {}).get('psi_minor', 0.1)
        self.psi_major_threshold = self.config.get('drift_thresholds', {}).get('psi_major', 0.25)
    
    def calculate_psi(self, feature):
        """Calculate Population Stability Index (PSI) for a feature."""
        try:
            # Get min/max bounds for binning
            min_val = min(
                self.reference_df[feature].min(),
                self.current_df[feature].min()
            )
            max_val = max(
                self.reference_df[feature].max(),
                self.current_df[feature].max()
            )
            
            # Create bins - use 10 bins or fewer if there are fewer unique values
            unique_vals = set(self.reference_df[feature].unique()) | set(self.current_df[feature].unique())
            n_bins = min(10, len(unique_vals))
            bins = np.linspace(min_val, max_val, n_bins+1)
            
            # Get bin frequencies
            ref_counts, _ = np.histogram(self.reference_df[feature], bins=bins)
            cur_counts, _ = np.histogram(self.current_df[feature], bins=bins)
            
            # Convert to percentages
            ref_pct = ref_counts / len(self.reference_df)
            cur_pct = cur_counts / len(self.current_df)
            
            # Replace zeros with a small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_pct = np.maximum(ref_pct, epsilon)
            cur_pct = np.maximum(cur_pct, epsilon)
            
            # Calculate PSI per bin and sum
            psi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
            psi = np.sum(psi_values)
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI for feature {feature}: {e}")
            return np.nan
    
    def run_ks_test(self, feature):
        """Run Kolmogorov-Smirnov test to detect distribution changes."""
        from scipy.stats import ks_2samp
        
        try:
            # Run KS test
            ks_stat, p_value = ks_2samp(
                self.reference_df[feature].dropna(), 
                self.current_df[feature].dropna()
            )
            return ks_stat, p_value
            
        except Exception as e:
            logger.error(f"Error running KS test for feature {feature}: {e}")
            return np.nan, np.nan
    
    def get_drift_severity(self, psi_value):
        """Determine drift severity based on PSI value."""
        if pd.isna(psi_value):
            return "Unknown"
        if psi_value > self.psi_major_threshold:
            return "Major drift"
        elif psi_value > self.psi_minor_threshold:
            return "Minor drift"
        else:
            return "No drift"
    
    def get_drift_report(self, features=None):
        """Generate a comprehensive drift report."""
        if features is None:
            # Use all common columns except target variable
            features = [col for col in self.reference_df.columns.intersection(self.current_df.columns)
                       if col != 'Class']
        
        results = []
        
        for feature in features:
            if feature not in self.reference_df.columns or feature not in self.current_df.columns:
                logger.warning(f"Feature {feature} not found in both datasets. Skipping.")
                continue
                
            # Calculate statistics
            ref_mean = self.reference_df[feature].mean()
            cur_mean = self.current_df[feature].mean()
            mean_diff_pct = (abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-10)) * 100
            
            ref_std = self.reference_df[feature].std()
            cur_std = self.current_df[feature].std()
            std_diff_pct = (abs(cur_std - ref_std) / (abs(ref_std) + 1e-10)) * 100
            
            # Calculate PSI
            psi = self.calculate_psi(feature)
            drift_severity_psi = self.get_drift_severity(psi)
            
            # Run KS test
            ks_stat, p_value = self.run_ks_test(feature)
            drift_detected_ks = bool(p_value < self.ks_threshold) if not pd.isna(p_value) else None
            
            # Store results
            results.append({
                'feature': feature,
                'ref_mean': ref_mean,
                'cur_mean': cur_mean,
                'mean_diff_pct': mean_diff_pct,
                'ref_std': ref_std,
                'cur_std': cur_std,
                'std_diff_pct': std_diff_pct,
                'psi': psi,
                'drift_severity_psi': drift_severity_psi,
                'ks_stat': ks_stat,
                'ks_p_value': p_value,
                'drift_detected_ks': drift_detected_ks
            })
        
        return pd.DataFrame(results)


class PerformanceMonitor:
    """Class to monitor model performance metrics."""
    
    def __init__(self, model_path, y_true, y_pred, y_prob=None, config=None):
        """Initialize with model information and predictions."""
        self.model_path = model_path
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.config = config or {}
        
        # Set thresholds from config or use defaults
        self.warning_threshold = self.config.get('performance_alert_thresholds', {}).get('warning_degradation_pct', 5.0)
        self.critical_threshold = self.config.get('performance_alert_thresholds', {}).get('critical_degradation_pct', 10.0)
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, average_precision_score
        )
        
        metrics = {}
        
        # Calculate basic metrics
        metrics['accuracy'] = float(accuracy_score(self.y_true, self.y_pred))
        metrics['precision'] = float(precision_score(self.y_true, self.y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(self.y_true, self.y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(self.y_true, self.y_pred, zero_division=0))
        
        # Calculate AUC metrics if probability scores available
        if self.y_prob is not None:
            metrics['roc_auc'] = float(roc_auc_score(self.y_true, self.y_prob))
            metrics['pr_auc'] = float(average_precision_score(self.y_true, self.y_prob))
        
        return metrics
    
    def compare_with_baseline(self, baseline_metrics):
        """Compare current metrics with baseline."""
        current_metrics = self.calculate_metrics()
        comparison = {}
        
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric)
            
            if baseline_value is None:
                logger.warning(f"Baseline for metric {metric} not found. Skipping comparison.")
                continue
            
            degradation_pct = ((baseline_value - current_value) / baseline_value) * 100
            
            # Determine status
            if degradation_pct > self.critical_threshold:
                status = 'CRITICAL'
            elif degradation_pct > self.warning_threshold:
                status = 'WARNING'
            else:
                status = 'OK'
            
            comparison[metric] = {
                'baseline': float(baseline_value),
                'current': float(current_value),
                'degradation_pct': float(degradation_pct),
                'status': status
            }
        
        return comparison


def generate_visualizations(reference_df, current_df, drift_report, metrics, output_dir):
    """Generate visualizations for monitoring report."""
    logger.info("Generating visualizations...")
    
    # 1. Feature distributions comparison (for top drifted features)
    top_drifted = drift_report.sort_values('psi', ascending=False).head(3)['feature'].tolist()
    for feature in top_drifted:
        plt.figure(figsize=(10, 6))
        
        # Create density plots
        sns.kdeplot(reference_df[feature], label="Reference", fill=True, alpha=0.3)
        sns.kdeplot(current_df[feature], label="Current", fill=True, alpha=0.3)
        
        plt.title(f"Distribution Comparison: {feature}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dist_{feature}.png"))
        plt.close()
    
    # 2. PSI values bar chart
    plt.figure(figsize=(12, 8))
    drift_report_sorted = drift_report.sort_values('psi', ascending=False)
    colors = ['red' if x == 'Major drift' else 'orange' if x == 'Minor drift' else 'green' 
             for x in drift_report_sorted['drift_severity_psi']]
    
    plt.bar(drift_report_sorted['feature'], drift_report_sorted['psi'], color=colors)
    plt.axhline(y=0.1, color='orange', linestyle='--', label='Minor Drift Threshold')
    plt.axhline(y=0.25, color='red', linestyle='--', label='Major Drift Threshold')
    plt.title('Feature Drift - Population Stability Index')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('PSI Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_drift_psi.png"))
    plt.close()
    
    # 3. Performance metrics comparison radar chart
    if metrics:
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot metrics
        angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        metric_values += metric_values[:1]  # Close the polygon
        
        ax.plot(angles, metric_values, 'o-', linewidth=2)
        ax.fill(angles, metric_values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Metrics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
        plt.close()
    
    # 4. Data overview comparison
    plt.figure(figsize=(10, 5))
    
    # Basic stats
    stats = [
        ('Sample Size', len(reference_df), len(current_df)),
        ('Fraud Rate (%)', reference_df['Class'].mean() * 100 if 'Class' in reference_df else 0, 
                       current_df['Class'].mean() * 100 if 'Class' in current_df else 0)
    ]
    
    x = np.arange(len(stats))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, [stat[1] for stat in stats], width, label='Reference')
    rects2 = ax.bar(x + width/2, [stat[2] for stat in stats], width, label='Current')
    
    ax.set_title('Dataset Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([stat[0] for stat in stats])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height) if height < 100 else '{:.0f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_comparison.png"))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


@click.command()
@click.option('--config-path', type=click.Path(exists=True), default='configs/monitoring.yaml', help='Path to the monitoring configuration file.')
@click.option('--reference-data-path', type=click.Path(exists=True), required=True, help='Path to the reference dataset (e.g., training data).')
@click.option('--current-data-path', type=click.Path(exists=True), required=True, help='Path to the current dataset to monitor.')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to the trained model file.')
@click.option('--output-dir', type=click.Path(), default='monitoring_reports', help='Directory to save monitoring reports.')
def run_monitoring(config_path, reference_data_path, current_data_path, model_path, output_dir):
    """
    Runs data drift and model performance monitoring.
    """
    logger.info("Starting monitoring process...")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    features_to_monitor = config.get('features_to_monitor', None)
    baseline_metrics = config.get('baseline_metrics', {})

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Load Data ---
    logger.info("Loading data...")
    reference_df = pd.read_csv(reference_data_path)
    current_df = pd.read_csv(current_data_path)

    # --- Data Drift Monitoring ---
    logger.info("Running data drift analysis...")
    drift_monitor = DataDriftMonitor(reference_df, current_df, config)
    drift_report = drift_monitor.get_drift_report(features=features_to_monitor)
    
    drift_report_path = os.path.join(output_dir, f"drift_report_{report_timestamp}.csv")
    drift_report.to_csv(drift_report_path, index=False)
    logger.info(f"Data drift report saved to {drift_report_path}")
    
    print("\n--- Data Drift Summary ---")
    print(drift_report)
    print("-" * 26)

    # --- Model Performance Monitoring ---
    logger.info("Running model performance analysis...")
    model = joblib.load(model_path)
    
    # Prepare data for prediction
    if 'Class' in current_df.columns:
        X_current = current_df.drop('Class', axis=1)
        y_true = current_df['Class']
    else:
        # If no ground truth, we can only check prediction drift, not performance
        logger.warning("No 'Class' column in current data. Performance metrics cannot be calculated.")
        return

    # Make predictions
    y_pred = pd.Series(model.predict(X_current), index=X_current.index)
    y_prob = pd.Series(model.predict_proba(X_current)[:, 1], index=X_current.index) if hasattr(model, 'predict_proba') else None

    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(model_path, y_true, y_pred, y_prob, config)
    
    # Calculate current metrics
    current_metrics = perf_monitor.calculate_metrics()
    
    # Compare with baseline
    comparison_report = perf_monitor.compare_with_baseline(baseline_metrics)

    # Save and print performance report
    perf_report = {
        'timestamp': datetime.now().isoformat(),
        'current_metrics': current_metrics,
        'baseline_comparison': comparison_report
    }
    
    # Generate visualizations if configured
    if config.get('reporting', {}).get('include_visualizations', False):
        plot_dir = os.path.join(output_dir, f"plots_{report_timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        generate_visualizations(reference_df, current_df, drift_report, current_metrics, plot_dir)
    
    # Save predictions if configured
    if config.get('reporting', {}).get('store_predictions', False):
        pred_df = pd.DataFrame({
            'true_value': y_true,
            'prediction': y_pred,
            'probability': y_prob if y_prob is not None else np.nan
        })
        pred_path = os.path.join(output_dir, f"predictions_{report_timestamp}.csv")
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")
    
    # Save performance report
    for format in config.get('reporting', {}).get('output_format', ['yaml']):
        if format.lower() == 'yaml':
            perf_report_path = os.path.join(output_dir, f"performance_report_{report_timestamp}.yaml")
            with open(perf_report_path, 'w') as f:
                yaml.dump(perf_report, f, default_flow_style=False)
        elif format.lower() == 'json':
            perf_report_path = os.path.join(output_dir, f"performance_report_{report_timestamp}.json")
            with open(perf_report_path, 'w') as f:
                json.dump(perf_report, f, indent=4)
    
    logger.info(f"Performance report saved to {output_dir}")

    print("\n--- Performance Summary ---")
    for metric, values in comparison_report.items():
        print(f"  {metric.capitalize()}:")
        print(f"    Current: {values['current']:.4f}, Baseline: {values['baseline']:.4f}")
        print(f"    Status: {values['status']}")
    print("-" * 29)

    # Create summary report with high-level metrics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_drift': {
            'features_monitored': len(drift_report),
            'features_with_drift': len(drift_report[drift_report['drift_severity_psi'] != 'No drift']),
            'major_drift_features': len(drift_report[drift_report['drift_severity_psi'] == 'Major drift']),
            'top_drifted_feature': drift_report.loc[drift_report['psi'].idxmax(), 'feature']
                                  if len(drift_report) > 0 else None
        },
        'performance': {
            'current_pr_auc': current_metrics.get('pr_auc'),
            'baseline_pr_auc': baseline_metrics.get('pr_auc'),
            'current_f1': current_metrics.get('f1_score'),
            'baseline_f1': baseline_metrics.get('f1_score')
        },
        'alerts': [f"{metric}: {values['status']}" 
                  for metric, values in comparison_report.items() 
                  if values['status'] != 'OK']
    }
    
    summary_path = os.path.join(output_dir, f"summary_{report_timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info("Monitoring process finished.")


if __name__ == '__main__':
    run_monitoring()