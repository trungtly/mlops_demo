from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')


class BaseFraudModel(ABC):
    """Abstract base class for fraud detection models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.model_metadata = {}
        
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying sklearn model."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'BaseFraudModel':
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional arguments for model fitting
        
        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.model = self._create_model()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit the model
        if X_val is not None and y_val is not None:
            # Use validation set if provided (for models that support it)
            if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
                self.model.fit(X, y, eval_set=[(X_val, y_val)], **kwargs)
            else:
                self.model.fit(X, y, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        
        # Store model metadata
        self.model_metadata = {
            'model_name': self.get_model_name(),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision_function
            decision_scores = self.model.decision_function(X)
            # Convert to probabilities using sigmoid
            proba_positive = 1 / (1 + np.exp(-decision_scores))
            return np.column_stack([1 - proba_positive, proba_positive])
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        
        if importance is not None:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return None
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'pr_auc': average_precision_score(y, y_proba),
            'accuracy': np.mean(y_pred == y),
            'precision': np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0,
            'recall': np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1) if np.sum(y == 1) > 0 else 0,
            'f1': 2 * (np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1)) * (np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1)) / ((np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1)) + (np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1))) if (np.sum(y_pred == 1) > 0 and np.sum(y == 1) > 0) else 0
        }
        
        return metrics
    
    def save_model(self, filepath: str, save_mlflow: bool = True) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib
        joblib.dump({
            'model': self.model,
            'metadata': self.model_metadata,
            'config': self.config
        }, filepath)
        
        # Save to MLflow if enabled
        if save_mlflow and mlflow.active_run():
            mlflow.sklearn.log_model(
                self.model, 
                f"models/{self.get_model_name().lower()}",
                registered_model_name=f"fraud_detection_{self.get_model_name().lower()}"
            )
    
    def load_model(self, filepath: str) -> 'BaseFraudModel':
        """Load model from disk."""
        saved_data = joblib.load(filepath)
        
        self.model = saved_data['model']
        self.model_metadata = saved_data['metadata']
        self.config = saved_data['config']
        self.feature_names = self.model_metadata.get('feature_names')
        self.is_fitted = True
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_name': self.get_model_name(),
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        
        if self.is_fitted:
            info.update(self.model_metadata)
            
            # Add model-specific parameters
            if hasattr(self.model, 'get_params'):
                info['parameters'] = self.model.get_params()
        
        return info
    
    def set_threshold(self, threshold: float) -> None:
        """Set prediction threshold for binary classification."""
        self.config['threshold'] = threshold
    
    def predict_with_threshold(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Make predictions with custom threshold."""
        proba = self.predict_proba(X)[:, 1]
        thresh = threshold or self.config.get('threshold', 0.5)
        return (proba >= thresh).astype(int)
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.get_model_name()}({status})"


class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, base_path: str = "models/"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.models = {}
    
    def register_model(self, name: str, model: BaseFraudModel) -> None:
        """Register a trained model."""
        if not model.is_fitted:
            raise ValueError("Only fitted models can be registered")
        
        self.models[name] = model
        
        # Save model to disk
        model_path = self.base_path / f"{name}.pkl"
        model.save_model(model_path, save_mlflow=False)
    
    def get_model(self, name: str) -> BaseFraudModel:
        """Get a registered model."""
        if name in self.models:
            return self.models[name]
        
        # Try to load from disk
        model_path = self.base_path / f"{name}.pkl"
        if model_path.exists():
            # We need to determine the model type to load it properly
            # For now, return None and require explicit loading
            raise ValueError(f"Model {name} found on disk but not in memory. Use load_model() explicitly.")
        
        raise ValueError(f"Model {name} not found")
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        disk_models = [f.stem for f in self.base_path.glob("*.pkl")]
        memory_models = list(self.models.keys())
        
        return list(set(disk_models + memory_models))
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare performance of all registered models."""
        results = []
        
        for name, model in self.models.items():
            if model.is_fitted:
                metrics = model.evaluate(X, y)
                metrics['model_name'] = name
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def get_best_model(self, X: pd.DataFrame, y: pd.Series, metric: str = 'pr_auc') -> Tuple[str, BaseFraudModel]:
        """Get the best model based on a specific metric."""
        comparison = self.compare_models(X, y)
        
        if comparison.empty:
            raise ValueError("No fitted models available for comparison")
        
        best_idx = comparison[metric].idxmax()
        best_model_name = comparison.loc[best_idx, 'model_name']
        
        return best_model_name, self.models[best_model_name]