#!/usr/bin/env python3
"""
Training script for fraud detection models.
Usage: python train_model.py --config configs/training.yaml --model xgboost
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from fraud_detection.data.ingestion import FraudDataIngestion
from fraud_detection.data.preprocessing import FraudDataPreprocessor
from fraud_detection.features.engineering import FraudFeatureEngineer
from fraud_detection.features.selection import FraudFeatureSelector
from fraud_detection.models import (
    create_ensemble_model, create_neural_network_model, ModelRegistry
)
from fraud_detection.evaluation.metrics import FraudMetrics
from fraud_detection.training.train import FraudTrainer
from fraud_detection.config import load_config


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/training.yaml',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['random_forest', 'xgboost', 'lightgbm', 'voting', 'stacking', 'mlp', 'deep_nn'],
        default='xgboost',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='fraud_detection',
        help='MLflow experiment name'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/creditcard.csv',
        help='Path to dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--feature-engineering',
        choices=['baseline', 'advanced'],
        default='baseline',
        help='Feature engineering complexity'
    )
    
    parser.add_argument(
        '--feature-selection',
        choices=['none', 'univariate', 'rfe', 'ensemble'],
        default='ensemble',
        help='Feature selection method'
    )
    
    parser.add_argument(
        '--hyperparameter-tuning',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    
    parser.add_argument(
        '--cross-validation',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def load_training_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using default configuration.")
        return {}


def main():
    """Main training function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting fraud detection model training")
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        try:
            # Log parameters
            mlflow.log_params({
                'model_type': args.model,
                'feature_engineering': args.feature_engineering,
                'feature_selection': args.feature_selection,
                'hyperparameter_tuning': args.hyperparameter_tuning,
                'cross_validation': args.cross_validation,
                'test_size': args.test_size,
                'random_state': args.random_state
            })
            
            # Initialize components
            logger.info("Initializing data components")
            data_ingestion = FraudDataIngestion()
            preprocessor = FraudDataPreprocessor()
            feature_engineer = FraudFeatureEngineer()
            feature_selector = FraudFeatureSelector()
            metrics_calculator = FraudMetrics()
            model_registry = ModelRegistry(args.output_dir)
            
            # Load and preprocess data
            logger.info(f"Loading data from {args.data_path}")
            if Path(args.data_path).exists():
                df = data_ingestion.load_local_data(args.data_path)
            else:
                logger.info("Local data not found. Downloading from Kaggle...")
                df = data_ingestion.download_creditcard_data()
            
            logger.info(f"Dataset shape: {df.shape}")
            mlflow.log_metric("dataset_size", df.shape[0])
            mlflow.log_metric("n_features", df.shape[1])
            mlflow.log_metric("fraud_rate", df['Class'].mean())
            
            # Preprocess data
            logger.info("Preprocessing data")
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
                df, test_size=args.test_size, random_state=args.random_state
            )
            
            # Feature engineering
            logger.info(f"Applying {args.feature_engineering} feature engineering")
            if args.feature_engineering == 'baseline':
                X_train = feature_engineer.engineer_features(
                    X_train, include_rolling=False, include_interactions=False
                )
                X_val = feature_engineer.engineer_features(
                    X_val, include_rolling=False, include_interactions=False
                )
                X_test = feature_engineer.engineer_features(
                    X_test, include_rolling=False, include_interactions=False
                )
            else:  # advanced
                X_train = feature_engineer.engineer_features(X_train)
                X_val = feature_engineer.engineer_features(X_val)
                X_test = feature_engineer.engineer_features(X_test)
            
            logger.info(f"Features after engineering: {X_train.shape[1]}")
            mlflow.log_metric("n_features_engineered", X_train.shape[1])
            
            # Feature selection
            if args.feature_selection != 'none':
                logger.info(f"Applying {args.feature_selection} feature selection")
                X_train_selected = feature_selector.select_features(
                    X_train, y_train, method=args.feature_selection
                )
                
                # Apply same selection to validation and test sets
                selected_features = X_train_selected.columns.tolist()
                X_val = X_val[selected_features]
                X_test = X_test[selected_features]
                X_train = X_train_selected
                
                logger.info(f"Features after selection: {X_train.shape[1]}")
                mlflow.log_metric("n_features_selected", X_train.shape[1])
            
            # Create model
            logger.info(f"Creating {args.model} model")
            model_config = config.get('models', {}).get(args.model, {})
            
            if args.model in ['random_forest', 'xgboost', 'lightgbm', 'voting', 'stacking']:
                model = create_ensemble_model(args.model, model_config)
            elif args.model in ['mlp', 'deep_nn']:
                model = create_neural_network_model(args.model, model_config)
            else:
                raise ValueError(f"Unknown model type: {args.model}")
            
            # Initialize trainer
            trainer = FraudTrainer(
                enable_hyperparameter_tuning=args.hyperparameter_tuning,
                cv_folds=args.cross_validation,
                random_state=args.random_state
            )
            
            # Train model
            logger.info("Training model")
            trained_model = trainer.train_model(
                model, X_train, y_train, X_val, y_val
            )
            
            # Evaluate model
            logger.info("Evaluating model")
            
            # Training metrics
            train_metrics = metrics_calculator.calculate_metrics(
                y_train, trained_model.predict_proba(X_train)[:, 1]
            )
            
            # Validation metrics
            val_metrics = metrics_calculator.calculate_metrics(
                y_val, trained_model.predict_proba(X_val)[:, 1]
            )
            
            # Test metrics
            test_metrics = metrics_calculator.calculate_metrics(
                y_test, trained_model.predict_proba(X_test)[:, 1]
            )
            
            # Log metrics
            for split, metrics in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{split}_{metric_name}", value)
            
            # Log feature importance if available
            feature_importance = trained_model.get_feature_importance()
            if feature_importance is not None:
                importance_path = f"feature_importance_{args.model}.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            
            # Save model
            model_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = Path(args.output_dir) / f"{model_name}.pkl"
            
            logger.info(f"Saving model to {model_path}")
            trained_model.save_model(model_path)
            
            # Register model
            model_registry.register_model(model_name, trained_model)
            
            # Log model artifact
            mlflow.log_artifact(str(model_path))
            
            # Print results
            logger.info("Training completed successfully!")
            logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
            logger.info(f"Test PR AUC: {test_metrics['pr_auc']:.4f}")
            logger.info(f"Test Recall@1%FPR: {test_metrics.get('recall_at_1_fpr', 'N/A')}")
            
            print("\n" + "="*50)
            print("TRAINING RESULTS")
            print("="*50)
            print(f"Model: {args.model}")
            print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
            print(f"Test PR AUC: {test_metrics['pr_auc']:.4f}")
            print(f"Test Precision: {test_metrics['precision']:.4f}")
            print(f"Test Recall: {test_metrics['recall']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print(f"Model saved to: {model_path}")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise
        else:
            mlflow.log_param("status", "success")


if __name__ == "__main__":
    main()