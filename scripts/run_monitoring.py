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

from fraud_detection.monitoring.drift import DataDriftMonitor
from fraud_detection.monitoring.performance import PerformanceMonitor

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
    click.echo("Starting monitoring process...")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    features_to_monitor = config.get('features_to_monitor', None)
    baseline_metrics = config.get('baseline_metrics', {})

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Load Data ---
    click.echo("Loading data...")
    reference_df = pd.read_csv(reference_data_path)
    current_df = pd.read_csv(current_data_path)

    # --- Data Drift Monitoring ---
    click.echo("Running data drift analysis...")
    drift_monitor = DataDriftMonitor(reference_df, current_df)
    drift_report = drift_monitor.get_drift_report(features=features_to_monitor)
    
    drift_report_path = os.path.join(output_dir, f"drift_report_{report_timestamp}.csv")
    drift_report.to_csv(drift_report_path, index=False)
    click.echo(f"Data drift report saved to {drift_report_path}")
    print("\n--- Data Drift Summary ---")
    print(drift_report)
    print("-" * 26)

    # --- Model Performance Monitoring ---
    click.echo("\nRunning model performance analysis...")
    model = joblib.load(model_path)
    
    # Prepare data for prediction
    if 'Class' in current_df.columns:
        X_current = current_df.drop('Class', axis=1)
        y_true = current_df['Class']
    else:
        # If no ground truth, we can only check prediction drift, not performance
        click.echo("Warning: No 'Class' column in current data. Performance metrics cannot be calculated.")
        return

    # Make predictions
    y_pred = pd.Series(model.predict(X_current), index=X_current.index)
    y_prob = pd.Series(model.predict_proba(X_current)[:, 1], index=X_current.index) if hasattr(model, 'predict_proba') else None

    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(model_path, y_true, y_pred, y_prob)
    
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
    
    perf_report_path = os.path.join(output_dir, f"performance_report_{report_timestamp}.yaml")
    with open(perf_report_path, 'w') as f:
        yaml.dump(perf_report, f, default_flow_style=False)
    click.echo(f"Performance report saved to {perf_report_path}")

    print("\n--- Performance Summary ---")
    for metric, values in comparison_report.items():
        print(f"  {metric.capitalize()}:")
        print(f"    Current: {values['current']:.4f}, Baseline: {values['baseline']:.4f}")
        print(f"    Status: {values['status']}")
    print("-" * 29)

    click.echo("\nMonitoring process finished.")


def generate_visualizations(reference_df, current_df, drift_report, metrics, output_dir):
    """Generate visualizations for monitoring report."""
    click.echo("\nGenerating visualizations...")
    
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
    
    click.echo(f"Visualizations saved to {output_dir}")


if __name__ == '__main__':
    run_monitoring()
