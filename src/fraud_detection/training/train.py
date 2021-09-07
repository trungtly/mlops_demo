"""Training pipeline for fraud detection models."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import time
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.integration import OptunaSearchCV

from ..config import config
from ..data.ingestion import FraudDataIngestion
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..evaluation.metrics import FraudDetectionMetrics

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, 
                 experiment_name: str = None,
                 cost_fp: float = 1.0, 
                 cost_fn: float = 10.0,
                 enable_hyperopt: bool = False,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            experiment_name: MLflow experiment name
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
            enable_hyperopt: Enable hyperparameter optimization
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.enable_hyperopt = enable_hyperopt
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Set up MLflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        
        self.metrics_calculator = FraudDetectionMetrics(cost_fp, cost_fn)
        
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations."""
        return {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=self.random_state,
                    class_weight="balanced",
                    max_iter=1000
                ),
                "params": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "liblinear"
                },
                "hyperopt_params": {
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"],
                    "class_weight": ["balanced", None]
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1
                ),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 10,
                    "min_samples_leaf": 5
                },
                "hyperopt_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [6, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "class_weight": ["balanced", "balanced_subsample", None]
                }
            },
            "xgboost": {
                "model": xgb.XGBClassifier(
                    random_state=self.random_state,
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
                },
                "hyperopt_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "scale_pos_weight": [50, 75, 99, 150]
                }
            },
            "lightgbm": {
                "model": lgb.LGBMClassifier(
                    random_state=self.random_state,
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
                },
                "hyperopt_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "num_leaves": [15, 31, 63]
                }
            },
            "mlp": {
                "model": MLPClassifier(
                    random_state=self.random_state,
                    early_stopping=True,
                    max_iter=500
                ),
                "params": {
                    "hidden_layer_sizes": (100, 50),
                    "alpha": 0.0001,
                    "learning_rate_init": 0.001,
                    "activation": "relu"
                },
                "hyperopt_params": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 50, 25)],
                    "alpha": [0.00001, 0.0001, 0.001, 0.01],
                    "learning_rate_init": [0.0001, 0.001, 0.01],
                    "activation": ["relu", "tanh"]
                }
            }
        }
    
    def prepare_data(self, 
                    X: pd.DataFrame, 
                    y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for training."""
        # Remove target and time columns if present
        feature_cols = [col for col in X.columns 
                       if col not in [config.TARGET_COLUMN, config.TIME_COLUMN]]
        
        X_processed = X[feature_cols]
        y_processed = y.values
        
        logger.info(f"Prepared data: X shape {X_processed.shape}, y shape {y_processed.shape}")
        return X_processed, y_processed
    
    def tune_hyperparameters(self,
                            model_name: str,
                            model: Any,
                            X_train: pd.DataFrame,
                            y_train: np.ndarray,
                            X_val: pd.DataFrame,
                            y_val: np.ndarray,
                            method: str = 'grid') -> Tuple[Any, Dict[str, Any]]:
        """
        Tune hyperparameters using grid search or Optuna.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            method: Hyperparameter tuning method ('grid' or 'optuna')
            
        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info(f"Tuning hyperparameters for {model_name} using {method}")
        start_time = time.time()
        
        # Get hyperparameter grid
        model_configs = self.get_model_configs()
        hyperopt_params = model_configs[model_name].get("hyperopt_params", {})
        
        if not hyperopt_params:
            logger.warning(f"No hyperparameters defined for {model_name}. Skipping tuning.")
            return model, {}
        
        # Define custom scorer for PR AUC
        pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
        
        # Define cross-validation strategy
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                             random_state=self.random_state)
        
        best_model = None
        best_params = {}
        
        if method == 'grid':
            # Grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=hyperopt_params,
                scoring=pr_auc_scorer,
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
        elif method == 'optuna':
            # Optuna search
            optuna_search = OptunaSearchCV(
                estimator=model,
                param_distributions=hyperopt_params,
                cv=cv,
                n_trials=100,
                scoring=pr_auc_scorer,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            optuna_search.fit(X_train, y_train)
            best_model = optuna_search.best_estimator_
            best_params = optuna_search.best_params_
            best_score = optuna_search.best_score_
        
        duration = time.time() - start_time
        logger.info(f"Hyperparameter tuning completed in {duration:.2f} seconds")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        # Evaluate on validation set
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, y_val_pred_proba)
        logger.info(f"Validation PR AUC with best params: {val_pr_auc:.4f}")
        
        # Log the hyperparameter tuning information
        mlflow.log_metric("hyperopt_duration_seconds", duration)
        mlflow.log_metric("hyperopt_best_cv_score", best_score)
        mlflow.log_metric("hyperopt_val_pr_auc", val_pr_auc)
        
        return best_model, best_params
    
    def train_model(self, 
                   model_name: str,
                   train_data: pd.DataFrame,
                   val_data: pd.DataFrame,
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
            # Log dataset information
            mlflow.log_param("train_size", len(X_train_processed))
            mlflow.log_param("val_size", len(X_val_processed))
            mlflow.log_param("fraud_rate_train", y_train.mean())
            mlflow.log_param("fraud_rate_val", y_val.mean())
            mlflow.log_param("feature_count", X_train_processed.shape[1])
            
            # Hyperparameter optimization if enabled
            if self.enable_hyperopt:
                model, best_params = self.tune_hyperparameters(
                    model_name, model, X_train_processed, y_train_processed, 
                    X_val_processed, y_val_processed, method='optuna'
                )
                params.update(best_params)
            
            # Log parameters after potential hyperopt updates
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            
            # Create timestamp for tracking training start
            train_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            mlflow.log_param("train_start_time", train_start_time)
            
            # Train model
            train_start = time.time()
            model.fit(X_train_processed, y_train_processed)
            train_duration = time.time() - train_start
            mlflow.log_metric("train_duration_seconds", train_duration)
            
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
            
            # Log additional metrics (e.g., recall at specific FPR)
            for threshold_key, threshold_value in val_metrics.items():
                if threshold_key not in ["basic_metrics", "cost_metrics"] and not isinstance(threshold_value, dict):
                    mlflow.log_metric(f"val_{threshold_key}", threshold_value)
            
            # Save model artifacts
            model_dir = config.MODELS_DIR / run.info.run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f"{model_name}_{train_start_time}.pkl"
            joblib.dump(model, model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"{config.MODEL_NAME}_{model_name}"
            )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train_processed.columns.tolist()
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = model_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
                
                # Plot feature importance
                plt.figure(figsize=(10, 8))
                top_features = importance_df.head(20)
                plt.barh(top_features['feature'], top_features['importance'])
                plt.xlabel('Importance')
                plt.title(f'Top 20 Feature Importance - {model_name}')
                plt.tight_layout()
                
                importance_plot_path = model_dir / "feature_importance.png"
                plt.savefig(importance_plot_path)
                plt.close()
                mlflow.log_artifact(str(importance_plot_path))
            
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
            
            # Log the decision threshold and threshold metrics
            thresholds = self.metrics_calculator.find_optimal_thresholds(
                y_val_processed, y_val_pred_proba
            )
            
            mlflow.log_params({f"threshold_{k}": v for k, v in thresholds.items()})
            
            # Save validation predictions for further analysis
            preds_df = pd.DataFrame({
                'true_label': y_val_processed,
                'pred_proba': y_val_pred_proba,
                'pred_label': y_val_pred
            })
            preds_path = model_dir / "val_predictions.csv"
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(str(preds_path))
            
            # Log results to console
            logger.info(f"Model {model_name} training completed. Run ID: {run.info.run_id}")
            logger.info(f"Validation metrics:")
            logger.info(f"  ROC AUC: {val_metrics['basic_metrics']['roc_auc']:.4f}")
            logger.info(f"  PR AUC: {val_metrics['basic_metrics']['pr_auc']:.4f}")
            logger.info(f"  F1 Score: {val_metrics['basic_metrics']['f1_score']:.4f}")
            logger.info(f"  Recall@1%FPR: {val_metrics.get('recall_at_1pct_fpr', 'N/A')}")
            logger.info(f"  Model saved to: {model_path}")
            
            return {
                "run_id": run.info.run_id,
                "model": model,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model_path": str(model_path),
                "thresholds": thresholds
            }
    
    def create_ensemble_model(self,
                            base_models: Dict[str, Any],
                            ensemble_type: str = 'voting',
                            weights: Optional[List[float]] = None,
                            meta_model=None) -> Dict[str, Any]:
        """
        Create an ensemble model from base models.
        
        Args:
            base_models: Dictionary of base models {model_name: model_info}
            ensemble_type: Type of ensemble ('voting' or 'stacking')
            weights: List of weights for voting ensemble
            meta_model: Meta-estimator for stacking ensemble
            
        Returns:
            Dictionary with ensemble model information
        """
        logger.info(f"Creating {ensemble_type} ensemble model")
        
        # Extract models from results
        estimators = [(name, info["model"]) for name, info in base_models.items()]
        
        # Create ensemble model
        if ensemble_type == 'voting':
            model = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights,
                n_jobs=-1
            )
            ensemble_name = 'voting_ensemble'
        else:  # stacking
            if meta_model is None:
                meta_model = LogisticRegression(random_state=self.random_state)
                
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=self.cv_folds,
                n_jobs=-1
            )
            ensemble_name = 'stacking_ensemble'
        
        # Train and evaluate the ensemble
        # Get data from one of the base models (all should have the same train/val data)
        first_model_info = next(iter(base_models.values()))
        
        # Reconstruct training and validation data (assuming they're stored in MLflow artifacts)
        train_data = first_model_info.get("train_data")
        val_data = first_model_info.get("val_data")
        
        # Train the ensemble model
        result = self.train_model(
            ensemble_name,
            train_data,
            val_data
        )
        
        return result
    
    def train_multiple_models(self, 
                            train_data: pd.DataFrame,
                            val_data: pd.DataFrame,
                            models_to_train: Optional[list] = None,
                            create_ensemble: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and compare performance."""
        if models_to_train is None:
            models_to_train = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        
        results = {}
        
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name} model")
                result = self.train_model(model_name, train_data, val_data)
                results[model_name] = result
                
                # Store train/val data for potential ensemble creation
                results[model_name]["train_data"] = train_data
                results[model_name]["val_data"] = val_data
                
                logger.info(f"{model_name} - Val ROC AUC: {result['val_metrics']['basic_metrics']['roc_auc']:.4f}")
                logger.info(f"{model_name} - Val PR AUC: {result['val_metrics']['basic_metrics']['pr_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Create ensemble if requested and at least 2 models were trained
        if create_ensemble and len(results) >= 2:
            try:
                # Create voting ensemble
                voting_result = self.create_ensemble_model(results, ensemble_type='voting')
                results['voting_ensemble'] = voting_result
                
                # Create stacking ensemble
                stacking_result = self.create_ensemble_model(results, ensemble_type='stacking')
                results['stacking_ensemble'] = stacking_result
                
            except Exception as e:
                logger.error(f"Failed to create ensemble models: {e}")
        
        # Find best model based on validation PR AUC
        best_model_name = max(
            results.keys(), 
            key=lambda x: results[x]['val_metrics']['basic_metrics']['pr_auc']
        )
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best PR AUC: {results[best_model_name]['val_metrics']['basic_metrics']['pr_auc']:.4f}")
        
        return results
    
    def run_training_pipeline(self, 
                            data_path: Optional[str] = None,
                            models_to_train: Optional[list] = None,
                            feature_engineering: bool = True) -> Dict[str, Dict[str, Any]]:
        """Run complete training pipeline."""
        logger.info("Starting training pipeline")
        
        # Data ingestion
        if data_path:
            df = pd.read_csv(data_path)
        else:
            ingestor = FraudDataIngestion()
            df, _ = ingestor.run_full_ingestion()
        
        # Data preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(df, imbalance_method="none")
        
        # Feature engineering if requested
        if feature_engineering:
            engineer = FeatureEngineer()
            
            # Apply to train
            processed_data["train"] = engineer.create_features(
                processed_data["train"],
                is_training=True
            )
            
            # Apply to validation with same transformations
            processed_data["val"] = engineer.create_features(
                processed_data["val"],
                is_training=False
            )
            
            # Apply to test with same transformations
            processed_data["test"] = engineer.create_features(
                processed_data["test"],
                is_training=False
            )
        
        # Train models
        results = self.train_multiple_models(
            processed_data["train"], 
            processed_data["val"],
            models_to_train,
            create_ensemble=True
        )
        
        # Evaluate best model on test set
        best_model_name = max(
            results.keys(), 
            key=lambda x: results[x]['val_metrics']['basic_metrics']['pr_auc']
        )
        
        best_model = results[best_model_name]["model"]
        
        # Prepare test data
        X_test = processed_data["test"].drop(columns=[config.TARGET_COLUMN])
        y_test = processed_data["test"][config.TARGET_COLUMN]
        X_test_processed, y_test_processed = self.prepare_data(X_test, y_test)
        
        # Evaluate on test set
        y_test_pred = best_model.predict(X_test_processed)
        y_test_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
        
        test_metrics = self.metrics_calculator.comprehensive_evaluation(
            y_test_processed, y_test_pred, y_test_pred_proba
        )
        
        # Log test metrics for best model
        with mlflow.start_run(run_id=results[best_model_name]["run_id"]):
            for metric_name, metric_value in test_metrics["basic_metrics"].items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            for metric_name, metric_value in test_metrics["cost_metrics"].items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Log additional metrics
            for threshold_key, threshold_value in test_metrics.items():
                if threshold_key not in ["basic_metrics", "cost_metrics"] and not isinstance(threshold_value, dict):
                    mlflow.log_metric(f"test_{threshold_key}", threshold_value)
        
        results[best_model_name]["test_metrics"] = test_metrics
        
        logger.info("Training pipeline completed")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Test ROC AUC: {test_metrics['basic_metrics']['roc_auc']:.4f}")
        logger.info(f"Test PR AUC: {test_metrics['basic_metrics']['pr_auc']:.4f}")
        
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
    parser.add_argument("--hyperopt", action="store_true",
                       help="Enable hyperparameter optimization")
    parser.add_argument("--feature-engineering", action="store_true",
                       help="Enable advanced feature engineering")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    trainer = ModelTrainer(
        experiment_name=args.experiment_name,
        enable_hyperopt=args.hyperopt
    )
    
    results = trainer.run_training_pipeline(
        data_path=args.data_path,
        models_to_train=args.models,
        feature_engineering=args.feature_engineering
    )
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    print("\nValidation Results:")
    for model_name, result in results.items():
        val_metrics = result['val_metrics']['basic_metrics']
        print(f"{model_name}:")
        print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  PR AUC: {val_metrics['pr_auc']:.4f}")
        print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
        if 'recall_at_1pct_fpr' in result['val_metrics']:
            print(f"  Recall@1%FPR: {result['val_metrics']['recall_at_1pct_fpr']:.4f}")
        print(f"  Run ID: {result['run_id']}")
    
    # Print test results for the best model
    best_model_name = max(
        results.keys(), 
        key=lambda x: results[x]['val_metrics']['basic_metrics']['pr_auc']
    )
    
    if 'test_metrics' in results[best_model_name]:
        test_metrics = results[best_model_name]['test_metrics']['basic_metrics']
        print("\nBest Model Test Results:")
        print(f"Model: {best_model_name}")
        print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  PR AUC: {test_metrics['pr_auc']:.4f}")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        
    print("="*50)


if __name__ == "__main__":
    main()