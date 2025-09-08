import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from typing import Dict, Any, Optional
import joblib

class PerformanceMonitor:
    """
    A class to monitor the performance of a classification model.
    """

    def __init__(self, model_path: str, y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.Series] = None):
        """
        Initialize the performance monitor.

        Args:
            model_path (str): Path to the trained model file.
            y_true (pd.Series): True labels.
            y_pred (pd.Series): Predicted labels.
            y_prob (Optional[pd.Series]): Predicted probabilities for the positive class.
        """
        self.model = joblib.load(model_path)
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate a comprehensive set of performance metrics.

        Returns:
            Dict[str, float]: A dictionary of performance metrics.
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, zero_division=0),
        }

        if self.y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_prob)
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
            metrics['pr_auc'] = auc(recall, precision)

        return metrics

    def compare_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Compare current performance with baseline metrics.

        Args:
            baseline_metrics (Dict[str, float]): A dictionary of baseline performance metrics.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary comparing current and baseline metrics.
        """
        current_metrics = self.calculate_metrics()
        comparison_report = {}

        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics.get(metric)
            if current_value is not None:
                degradation = (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0
                comparison_report[metric] = {
                    'current': current_value,
                    'baseline': baseline_value,
                    'degradation_pct': degradation * 100,
                    'status': self._get_status(degradation)
                }
        return comparison_report

    @staticmethod
    def _get_status(degradation: float) -> str:
        """
        Determine the performance status based on degradation percentage.

        Args:
            degradation (float): The performance degradation percentage.

        Returns:
            str: The performance status ('OK', 'Warning', 'Critical').
        """
        if degradation > 10:
            return 'Critical'
        elif degradation > 5:
            return 'Warning'
        else:
            return 'OK'

if __name__ == '__main__':
    # Example Usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # 1. Create and save a dummy model
    X, y = pd.DataFrame(abs(np.random.randn(100, 5))), pd.Series(np.random.randint(0, 2, 100))
    model = RandomForestClassifier()
    model.fit(X, y)
    model_path = 'dummy_model.joblib'
    joblib.dump(model, model_path)

    # 2. Generate some test data
    X_test, y_test = pd.DataFrame(abs(np.random.randn(50, 5))), pd.Series(np.random.randint(0, 2, 50))
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Initialize the monitor
    monitor = PerformanceMonitor(model_path=model_path, y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # 4. Calculate current metrics
    current_performance = monitor.calculate_metrics()
    print("Current Model Performance:")
    for metric, value in current_performance.items():
        print(f"  {metric}: {value:.4f}")

    # 5. Compare with a baseline
    baseline = {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'f1_score': 0.77,
        'roc_auc': 0.90,
        'pr_auc': 0.85
    }
    
    comparison = monitor.compare_with_baseline(baseline_metrics=baseline)
    print("\nPerformance Comparison with Baseline:")
    for metric, values in comparison.items():
        print(f"  {metric.capitalize()}:")
        print(f"    Current: {values['current']:.4f}, Baseline: {values['baseline']:.4f}")
        print(f"    Degradation: {values['degradation_pct']:.2f}%")
        print(f"    Status: {values['status']}")

# Real-time model performance monitoring
