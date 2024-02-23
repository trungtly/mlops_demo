"""Evaluation metrics for fraud detection models."""

import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FraudDetectionMetrics:
    """Comprehensive metrics for fraud detection evaluation."""
    
    def __init__(self, cost_fp: float = 1.0, cost_fn: float = 10.0):
        """
        Initialize metrics calculator.
        
        Args:
            cost_fp: Cost of false positive (flagging legitimate transaction)
            cost_fn: Cost of false negative (missing fraud)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        
    def calculate_basic_metrics(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "specificity": self._calculate_specificity(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics.update({
                "roc_auc": roc_auc_score(y_true, y_pred_proba),
                "pr_auc": average_precision_score(y_true, y_pred_proba),
            })
        
        return metrics
    
    def calculate_cost_metrics(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate cost-based metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)
        avg_cost_per_transaction = total_cost / len(y_true)
        
        # Cost reduction compared to baseline (predict all as non-fraud)
        baseline_cost = sum(y_true) * self.cost_fn
        cost_reduction = (baseline_cost - total_cost) / baseline_cost if baseline_cost > 0 else 0
        
        return {
            "total_cost": total_cost,
            "avg_cost_per_transaction": avg_cost_per_transaction,
            "cost_reduction": cost_reduction,
            "false_positive_cost": fp * self.cost_fp,
            "false_negative_cost": fn * self.cost_fn,
        }
    
    def calculate_threshold_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_pred_proba: np.ndarray,
                                  thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Calculate metrics across different thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_pred_proba)
            cost_metrics = self.calculate_cost_metrics(y_true, y_pred)
            
            metrics_row = {
                "threshold": threshold,
                **basic_metrics,
                **cost_metrics
            }
            threshold_metrics.append(metrics_row)
        
        return pd.DataFrame(threshold_metrics)
    
    def find_optimal_threshold(self, 
                             y_true: np.ndarray, 
                             y_pred_proba: np.ndarray,
                             metric: str = "f1_score") -> Tuple[float, float]:
        """Find optimal threshold based on specified metric."""
        threshold_df = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        if metric == "cost":
            # Minimize total cost
            optimal_idx = threshold_df["total_cost"].idxmin()
        else:
            # Maximize the specified metric
            optimal_idx = threshold_df[metric].idxmax()
        
        optimal_threshold = threshold_df.loc[optimal_idx, "threshold"]
        optimal_value = threshold_df.loc[optimal_idx, metric]
        
        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} "
                   f"(value: {optimal_value:.4f})")
        
        return optimal_threshold, optimal_value
    
    def calculate_recall_at_fpr(self, 
                              y_true: np.ndarray, 
                              y_pred_proba: np.ndarray,
                              max_fpr: float = 0.01) -> float:
        """Calculate recall at maximum false positive rate."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Find the highest recall where FPR <= max_fpr
        valid_indices = fpr <= max_fpr
        if not any(valid_indices):
            return 0.0
        
        max_recall = tpr[valid_indices].max()
        logger.info(f"Recall at {max_fpr*100}% FPR: {max_recall:.4f}")
        
        return max_recall
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> str:
        """Generate detailed classification report."""
        return classification_report(
            y_true, y_pred, 
            target_names=["Legitimate", "Fraud"],
            digits=4
        )
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            normalize: bool = False,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        ax.set_xticklabels(['Legitimate', 'Fraud'])
        ax.set_yticklabels(['Legitimate', 'Fraud'])
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, 
               label=f'PR curve (AUC = {pr_auc:.4f})')
        
        # Baseline (random classifier)
        baseline = sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                  label=f'Baseline (AP = {baseline:.4f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               threshold: float = 0.5) -> Dict[str, Any]:
        """Run comprehensive evaluation and return all metrics."""
        
        # If predictions are probabilities, convert to binary with threshold
        if y_pred_proba is not None and len(np.unique(y_pred)) == 2:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred_binary = y_pred
            
        results = {}
        
        # Basic metrics
        results["basic_metrics"] = self.calculate_basic_metrics(
            y_true, y_pred_binary, y_pred_proba
        )
        
        # Cost-based metrics
        results["cost_metrics"] = self.calculate_cost_metrics(y_true, y_pred_binary)
        
        # Classification report
        results["classification_report"] = self.generate_classification_report(
            y_true, y_pred_binary
        )
        
        # Advanced metrics if probabilities available
        if y_pred_proba is not None:
            results["recall_at_1pct_fpr"] = self.calculate_recall_at_fpr(
                y_true, y_pred_proba, max_fpr=0.01
            )
            
            # Find optimal thresholds
            results["optimal_threshold_f1"], _ = self.find_optimal_threshold(
                y_true, y_pred_proba, "f1_score"
            )
            results["optimal_threshold_cost"], _ = self.find_optimal_threshold(
                y_true, y_pred_proba, "cost"
            )
        
        logger.info("Comprehensive evaluation completed")
        return results


def main():
    """Example usage of metrics."""
    # Generate example data
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=1000, p=[0.99, 0.01])
    y_pred_proba = np.random.random(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics_calculator = FraudDetectionMetrics(cost_fp=1, cost_fn=10)
    results = metrics_calculator.comprehensive_evaluation(
        y_true, y_pred, y_pred_proba
    )
    
    print("Basic Metrics:")
    for metric, value in results["basic_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nCost Metrics:")
    for metric, value in results["cost_metrics"].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
# Custom metrics for fraud detection evaluation
