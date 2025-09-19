from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import xgboost as xgb
import lightgbm as lgb
from .base import BaseFraudModel


class RandomForestFraudModel(BaseFraudModel):
    """Random Forest model for fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _create_model(self) -> BaseEstimator:
        return RandomForestClassifier(**self.config)
    
    def get_model_name(self) -> str:
        return "RandomForest"


class XGBoostFraudModel(BaseFraudModel):
    """XGBoost model for fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 100,  # Handle class imbalance
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'aucpr'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _create_model(self) -> BaseEstimator:
        return xgb.XGBClassifier(**self.config)
    
    def get_model_name(self) -> str:
        return "XGBoost"


class LightGBMFraudModel(BaseFraudModel):
    """LightGBM model for fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _create_model(self) -> BaseEstimator:
        return lgb.LGBMClassifier(**self.config)
    
    def get_model_name(self) -> str:
        return "LightGBM"


class GradientBoostingFraudModel(BaseFraudModel):
    """Gradient Boosting model for fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _create_model(self) -> BaseEstimator:
        return GradientBoostingClassifier(**self.config)
    
    def get_model_name(self) -> str:
        return "GradientBoosting"


class AdaBoostFraudModel(BaseFraudModel):
    """AdaBoost model for fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'n_estimators': 100,
            'learning_rate': 1.0,
            'random_state': 42
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _create_model(self) -> BaseEstimator:
        return AdaBoostClassifier(**self.config)
    
    def get_model_name(self) -> str:
        return "AdaBoost"


class VotingEnsembleFraudModel(BaseFraudModel):
    """Voting ensemble of multiple models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'voting': 'soft',  # Use probability averaging
            'n_jobs': -1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Define base models
        self.base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42,
                class_weight='balanced', n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                scale_pos_weight=100, random_state=42, n_jobs=-1,
                eval_metric='aucpr'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                class_weight='balanced', random_state=42, n_jobs=-1,
                verbose=-1
            ))
        ]
    
    def _create_model(self) -> BaseEstimator:
        return VotingClassifier(
            estimators=self.base_models,
            voting=self.config['voting'],
            n_jobs=self.config['n_jobs']
        )
    
    def get_model_name(self) -> str:
        return "VotingEnsemble"


class StackingEnsembleFraudModel(BaseFraudModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'cv': 5,
            'n_jobs': -1,
            'passthrough': False
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Define base models
        self.base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42,
                class_weight='balanced', n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                scale_pos_weight=100, random_state=42, n_jobs=-1,
                eval_metric='aucpr'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                class_weight='balanced', random_state=42, n_jobs=-1,
                verbose=-1
            ))
        ]
        
        # Meta-learner
        self.meta_learner = LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000
        )
    
    def _create_model(self) -> BaseEstimator:
        return StackingClassifier(
            estimators=self.base_models,
            final_estimator=self.meta_learner,
            cv=self.config['cv'],
            n_jobs=self.config['n_jobs'],
            passthrough=self.config['passthrough']
        )
    
    def get_model_name(self) -> str:
        return "StackingEnsemble"


class CustomWeightedEnsemble(BaseFraudModel):
    """Custom weighted ensemble with learned weights."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'models': ['rf', 'xgb', 'lgb'],
            'weight_learning_method': 'validation'  # 'validation' or 'uniform'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Initialize base models
        self.models = {}
        if 'rf' in self.config['models']:
            self.models['rf'] = RandomForestFraudModel()
        if 'xgb' in self.config['models']:
            self.models['xgb'] = XGBoostFraudModel()
        if 'lgb' in self.config['models']:
            self.models['lgb'] = LightGBMFraudModel()
        if 'gb' in self.config['models']:
            self.models['gb'] = GradientBoostingFraudModel()
        
        self.weights = None
    
    def _create_model(self) -> BaseEstimator:
        # This is a placeholder since we handle ensemble logic manually
        return None
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'CustomWeightedEnsemble':
        """Fit all base models and learn ensemble weights."""
        
        # Fit all base models
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            model.fit(X, y, X_val, y_val, **kwargs)
        
        # Learn ensemble weights
        if X_val is not None and y_val is not None and self.config['weight_learning_method'] == 'validation':
            self._learn_weights(X_val, y_val)
        else:
            # Use uniform weights
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        self.is_fitted = True
        self.feature_names = list(X.columns)
        
        # Store metadata
        self.model_metadata = {
            'model_name': self.get_model_name(),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_names': self.feature_names,
            'config': self.config,
            'weights': self.weights
        }
        
        return self
    
    def _learn_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Learn ensemble weights based on validation performance."""
        from scipy.optimize import minimize
        from sklearn.metrics import average_precision_score
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X_val)[:, 1]
        
        # Optimize weights to maximize PR AUC
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.zeros(len(y_val))
            
            for i, (name, pred) in enumerate(predictions.items()):
                ensemble_pred += weights[i] * pred
            
            return -average_precision_score(y_val, ensemble_pred)
        
        # Initial weights (uniform)
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Store learned weights
        self.weights = dict(zip(self.models.keys(), result.x))
        
        print(f"Learned weights: {self.weights}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        ensemble_pred = np.zeros((len(X), 2))
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            weight = self.weights[name]
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_model_name(self) -> str:
        return "CustomWeightedEnsemble"
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get weighted feature importance from all models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_dfs = []
        
        for name, model in self.models.items():
            model_importance = model.get_feature_importance()
            if model_importance is not None:
                model_importance['importance'] *= self.weights[name]
                model_importance['model'] = name
                importance_dfs.append(model_importance)
        
        if importance_dfs:
            # Combine importance from all models
            combined_df = pd.concat(importance_dfs)
            
            # Aggregate by feature
            aggregated = combined_df.groupby('feature')['importance'].sum().reset_index()
            return aggregated.sort_values('importance', ascending=False)
        
        return None


def create_ensemble_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseFraudModel:
    """Factory function to create ensemble models."""
    
    models = {
        'random_forest': RandomForestFraudModel,
        'xgboost': XGBoostFraudModel,
        'lightgbm': LightGBMFraudModel,
        'gradient_boosting': GradientBoostingFraudModel,
        'adaboost': AdaBoostFraudModel,
        'voting': VotingEnsembleFraudModel,
        'stacking': StackingEnsembleFraudModel,
        'weighted': CustomWeightedEnsemble
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](config)