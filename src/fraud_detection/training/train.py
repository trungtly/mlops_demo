"""Training pipeline for fraud detection models."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

from ..config import config
from ..data.ingestion import DataIngestion
from ..data.preprocessing import DataPreprocessor
from ..evaluation.metrics import FraudDetectionMetrics

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, 
                 experiment_name: str = None,
                 cost_fp: float = 1.0, 
                 cost_fn: float = 10.0):
        """
        Initialize trainer.
        
        Args:
            experiment_name: MLflow experiment name
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
        """
        self.experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        
        # Set up MLflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        
        self.metrics_calculator = FraudDetectionMetrics(cost_fp, cost_fn)
        
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations."""
        return {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=config.RANDOM_SEED,
                    class_weight="balanced",
                    max_iter=1000
                ),
                "params": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "liblinear"
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    random_state=config.RANDOM_SEED,
                    class_weight="balanced",
                    n_jobs=-1
                ),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 10,
                    "min_samples_leaf": 5
                }
            },
            "xgboost": {
                "model": xgb.XGBClassifier(
                    random_state=config.RANDOM_SEED,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    n_jobs=-1
                ),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": 99  # Adjust for class imbalance
                }
            },
            "lightgbm": {
                "model": lgb.LGBMClassifier(
                    random_state=config.RANDOM_SEED,
                    class_weight="balanced",
                    n_jobs=-1,
                    verbose=-1
                ),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "num_leaves": 31
                }
            }
        }
    
    def prepare_data(self, 
                    X: pd.DataFrame, 
                    y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Remove target and time columns if present
        feature_cols = [col for col in X.columns 
                       if col not in [config.TARGET_COLUMN, config.TIME_COLUMN]]
        
        X_processed = X[feature_cols].values
        y_processed = y.values
        
        logger.info(f"Prepared data: X shape {X_processed.shape}, y shape {y_processed.shape}")
        return X_processed, y_processed
    
    def train_model(self, 
                   model_name: str,
                   train_data: Dict[str, pd.DataFrame],
                   val_data: Dict[str, pd.DataFrame],
                   custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a single model."""
        logger.info(f"Training {model_name} model")
        
        # Get model configuration
        model_configs = self.get_model_configs()
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config_dict = model_configs[model_name]
        model = config_dict["model"]
        params = config_dict["params"].copy()
        
        # Update with custom parameters
        if custom_params:
            params.update(custom_params)
            model.set_params(**custom_params)
        
        # Prepare data
        X_train = train_data.drop(columns=[config.TARGET_COLUMN])
        y_train = train_data[config.TARGET_COLUMN]
        X_val = val_data.drop(columns=[config.TARGET_COLUMN])
        y_val = val_data[config.TARGET_COLUMN]
        
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        X_val_processed, y_val_processed = self.prepare_data(X_val, y_val)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_size", len(X_train_processed))
            mlflow.log_param("val_size", len(X_val_processed))
            mlflow.log_param("fraud_rate_train", y_train_processed.mean())
            mlflow.log_param("fraud_rate_val", y_val_processed.mean())
            
            # Train model
            model.fit(X_train_processed, y_train_processed)
            
            # Make predictions
            y_train_pred = model.predict(X_train_processed)
            y_train_pred_proba = model.predict_proba(X_train_processed)[:, 1]
            y_val_pred = model.predict(X_val_processed)
            y_val_pred_proba = model.predict_proba(X_val_processed)[:, 1]
            
            # Calculate metrics
            train_metrics = self.metrics_calculator.comprehensive_evaluation(
                y_train_processed, y_train_pred, y_train_pred_proba
            )
            val_metrics = self.metrics_calculator.comprehensive_evaluation(
                y_val_processed, y_val_pred, y_val_pred_proba
            )
            
            # Log metrics
            for metric_name, metric_value in train_metrics["basic_metrics"].items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            for metric_name, metric_value in val_metrics["basic_metrics"].items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            for metric_name, metric_value in train_metrics["cost_metrics"].items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            for metric_name, metric_value in val_metrics["cost_metrics"].items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            # Log additional metrics
            if "recall_at_1pct_fpr" in val_metrics:
                mlflow.log_metric("val_recall_at_1pct_fpr", val_metrics["recall_at_1pct_fpr"])
            
            # Save model artifacts
            model_dir = config.MODELS_DIR / run.info.run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"{config.MODEL_NAME}_{model_name}"
            )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_names = [col for col in X_train.columns 
                               if col not in [config.TARGET_COLUMN, config.TIME_COLUMN]]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = model_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
            
            # Generate and log plots
            plots_dir = model_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Confusion matrix
            cm_fig = self.metrics_calculator.plot_confusion_matrix(
                y_val_processed, y_val_pred, 
                save_path=plots_dir / "confusion_matrix.png"
            )
            mlflow.log_artifact(str(plots_dir / "confusion_matrix.png"))
            
            # ROC curve
            roc_fig = self.metrics_calculator.plot_roc_curve(
                y_val_processed, y_val_pred_proba,
                save_path=plots_dir / "roc_curve.png"
            )
            mlflow.log_artifact(str(plots_dir / "roc_curve.png"))
            
            # PR curve
            pr_fig = self.metrics_calculator.plot_precision_recall_curve(
                y_val_processed, y_val_pred_proba,
                save_path=plots_dir / "pr_curve.png"
            )
            mlflow.log_artifact(str(plots_dir / "pr_curve.png"))
            
            logger.info(f"Model {model_name} training completed. Run ID: {run.info.run_id}")
            
            return {
                "run_id": run.info.run_id,
                "model": model,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model_path": str(model_path)
            }
    
    def train_multiple_models(self, 
                            train_data: pd.DataFrame,
                            val_data: pd.DataFrame,
                            models_to_train: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and compare performance."""
        if models_to_train is None:
            models_to_train = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        
        results = {}
        
        for model_name in models_to_train:
            try:
                result = self.train_model(model_name, train_data, val_data)
                results[model_name] = result
                
                logger.info(f"{model_name} - Val ROC AUC: {result['val_metrics']['basic_metrics']['roc_auc']:.4f}")
                logger.info(f"{model_name} - Val PR AUC: {result['val_metrics']['basic_metrics']['pr_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Find best model based on validation PR AUC
        best_model_name = max(
            results.keys(), 
            key=lambda x: results[x]['val_metrics']['basic_metrics']['pr_auc']
        )
        
        logger.info(f"Best model: {best_model_name}")
        
        return results
    
    def run_training_pipeline(self, 
                            data_path: Optional[str] = None,
                            models_to_train: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """Run complete training pipeline."""
        logger.info("Starting training pipeline")
        
        # Data ingestion
        if data_path:
            df = pd.read_csv(data_path)
        else:
            ingestor = DataIngestion()
            df, _ = ingestor.run()
        
        # Data preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(df, imbalance_method="none")
        
        # Train models
        results = self.train_multiple_models(
            processed_data["train"], 
            processed_data["val"],
            models_to_train
        )
        
        logger.info("Training pipeline completed")
        return results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument("--models", nargs="+", 
                       default=["logistic_regression", "xgboost", "lightgbm"],
                       help="Models to train")
    parser.add_argument("--experiment-name", type=str, 
                       default="fraud_detection_experiment",
                       help="MLflow experiment name")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    trainer = ModelTrainer(experiment_name=args.experiment_name)
    results = trainer.run_training_pipeline(
        data_path=args.data_path,
        models_to_train=args.models
    )
    
    print("\nTraining Results:")
    for model_name, result in results.items():
        val_metrics = result['val_metrics']['basic_metrics']
        print(f"{model_name}:")
        print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  PR AUC: {val_metrics['pr_auc']:.4f}")
        print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"  Run ID: {result['run_id']}")


if __name__ == "__main__":
    main()