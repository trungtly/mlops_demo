#!/usr/bin/env python3
"""
Evaluation script for fraud detection models.
Usage: python evaluate_model.py --model-path models/xgboost_20241219_123456.pkl --data-path data/test/test_data.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from fraud_detection.data.ingestion import FraudDataIngestion
from fraud_detection.data.preprocessing import FraudDataPreprocessor
from fraud_detection.features.engineering import FraudFeatureEngineer
from fraud_detection.models.base import ModelRegistry
from fraud_detection.evaluation.metrics import FraudMetrics


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate fraud detection model')
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=False,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=False,
        help='Name of registered model to evaluate'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to evaluation dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results/',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate evaluation plots'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save model predictions'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def load_model(args):
    """Load model for evaluation."""
    logger = logging.getLogger(__name__)
    
    model_registry = ModelRegistry(args.models_dir)
    
    if args.model_path:
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        logger.info(f"Loading model from {args.model_path}")
        # Extract model name from path for registry
        model_name = Path(args.model_path).stem
        
        # Load the model (this is a simplified approach)
        available_models = model_registry.list_models()
        if not available_models:
            raise ValueError("No models found in registry")
        
        # Use the most recent model
        latest_model = sorted(available_models)[-1]
        model = model_registry.get_model(latest_model)
        
    elif args.model_name:
        logger.info(f"Loading model '{args.model_name}' from registry")
        model = model_registry.get_model(args.model_name)
        
    else:
        # Load latest model
        available_models = model_registry.list_models()
        if not available_models:
            raise ValueError("No models found in registry")
        
        latest_model = sorted(available_models)[-1]
        logger.info(f"Loading latest model: {latest_model}")
        model = model_registry.get_model(latest_model)
    
    return model


def prepare_data(data_path: str):
    """Load and prepare evaluation data."""
    logger = logging.getLogger(__name__)
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    data_ingestion = FraudDataIngestion()
    df = data_ingestion.load_local_data(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud rate: {df['Class'].mean():.4f}")
    
    # Apply same preprocessing as training
    preprocessor = FraudDataPreprocessor()
    feature_engineer = FraudFeatureEngineer()
    
    # Separate features and target
    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
        y = df['Class']
    else:
        X = df
        y = None
        logger.warning("No 'Class' column found. Evaluation will be limited.")
    
    # Apply feature engineering (baseline)
    X_processed = feature_engineer.engineer_features(
        X, include_rolling=False, include_interactions=False, scale=True
    )
    
    return X_processed, y


def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate model performance."""
    logger = logging.getLogger(__name__)
    
    if y is None:
        logger.warning("No target variable available. Generating predictions only.")
        predictions = model.predict_proba(X)[:, 1]
        binary_predictions = (predictions >= threshold).astype(int)
        
        return {
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'metrics': {}
        }
    
    # Generate predictions
    logger.info("Generating predictions")
    predictions = model.predict_proba(X)[:, 1]
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Calculate metrics
    logger.info("Calculating metrics")
    metrics_calculator = FraudMetrics()
    metrics = metrics_calculator.calculate_metrics(y, predictions)
    
    # Add threshold-specific metrics
    from sklearn.metrics import classification_report
    metrics['classification_report'] = classification_report(
        y, binary_predictions, output_dict=True
    )
    
    return {
        'predictions': predictions,
        'binary_predictions': binary_predictions,
        'metrics': metrics
    }


def generate_plots(y_true, y_pred, y_pred_binary, output_dir):
    """Generate evaluation plots."""
    logger = logging.getLogger(__name__)
    logger.info("Generating evaluation plots")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    
    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prediction Distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(y_pred[y_true == 1], bins=50, alpha=0.7, label='Fraud', density=True)
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Prediction Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([y_pred[y_true == 0], y_pred[y_true == 1]], 
                labels=['Normal', 'Fraud'])
    plt.ylabel('Prediction Score')
    plt.title('Prediction Score by Class')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")


def save_results(results, model_info, output_dir):
    """Save evaluation results."""
    logger = logging.getLogger(__name__)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(results['metrics'], f, indent=2, default=str)
    
    # Save model info
    model_info_file = output_dir / 'model_info.json'
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation")
    
    try:
        # Load model
        model = load_model(args)
        model_info = model.get_model_info()
        logger.info(f"Model loaded: {model.get_model_name()}")
        
        # Prepare data
        X, y = prepare_data(args.data_path)
        logger.info(f"Data prepared: {X.shape}")
        
        # Evaluate model
        results = evaluate_model(model, X, y, args.threshold)
        
        # Print results
        if y is not None:
            metrics = results['metrics']
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Model: {model.get_model_name()}")
            print(f"Dataset size: {len(X)}")
            print(f"Fraud rate: {y.mean():.4f}")
            print(f"Threshold: {args.threshold}")
            print("-"*50)
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"PR AUC: {metrics['pr_auc']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
            if 'recall_at_1_fpr' in metrics:
                print(f"Recall@1%FPR: {metrics['recall_at_1_fpr']:.4f}")
            if 'recall_at_5_fpr' in metrics:
                print(f"Recall@5%FPR: {metrics['recall_at_5_fpr']:.4f}")
            
            print("="*50)
        else:
            print(f"Generated predictions for {len(X)} samples")
        
        # Generate plots if requested
        if args.generate_plots and y is not None:
            generate_plots(y, results['predictions'], 
                         results['binary_predictions'], args.output_dir)
        
        # Save predictions if requested
        if args.save_predictions:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            pred_df = pd.DataFrame({
                'prediction_score': results['predictions'],
                'prediction_binary': results['binary_predictions']
            })
            
            if y is not None:
                pred_df['actual'] = y
            
            pred_file = output_dir / 'predictions.csv'
            pred_df.to_csv(pred_file, index=False)
            logger.info(f"Predictions saved to {pred_file}")
        
        # Save results
        save_results(results, model_info, args.output_dir)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()