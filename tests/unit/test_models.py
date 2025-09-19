import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from fraud_detection.models.base import BaseFraudModel, ModelRegistry
from fraud_detection.models.ensemble import (
    RandomForestFraudModel, XGBoostFraudModel, LightGBMFraudModel,
    VotingEnsembleFraudModel, StackingEnsembleFraudModel,
    CustomWeightedEnsemble, create_ensemble_model
)
from fraud_detection.models.neural_network import (
    MLPFraudModel, DeepNeuralNetworkFraudModel, 
    create_neural_network_model
)


class TestBaseFraudModel:
    """Test BaseFraudModel abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseFraudModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFraudModel()


class TestRandomForestFraudModel:
    """Test RandomForestFraudModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = RandomForestFraudModel()
        assert model.get_model_name() == "RandomForest"
        assert not model.is_fitted
    
    def test_init_with_config(self):
        """Test model initialization with custom config."""
        config = {'n_estimators': 50, 'max_depth': 5}
        model = RandomForestFraudModel(config)
        
        assert model.config['n_estimators'] == 50
        assert model.config['max_depth'] == 5
    
    def test_fit_predict(self, sample_fraud_data):
        """Test model fitting and prediction."""
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        model = RandomForestFraudModel({'n_estimators': 10, 'random_state': 42})
        
        # Fit model
        fitted_model = model.fit(X, y)
        assert fitted_model.is_fitted
        assert fitted_model.feature_names == list(X.columns)
        
        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probabilities
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_get_feature_importance(self, trained_rf_model):
        """Test feature importance extraction."""
        importance_df = trained_rf_model.get_feature_importance()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(trained_rf_model.feature_names)
        
        # Check that importances are sorted
        importances = importance_df['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    def test_evaluate(self, trained_rf_model, sample_fraud_data):
        """Test model evaluation."""
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        metrics = trained_rf_model.evaluate(X, y)
        
        expected_metrics = ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_save_load_model(self, trained_rf_model, temp_model_path):
        """Test model saving and loading."""
        # Save model
        trained_rf_model.save_model(str(temp_model_path), save_mlflow=False)
        assert temp_model_path.exists()
        
        # Load model
        new_model = RandomForestFraudModel()
        loaded_model = new_model.load_model(str(temp_model_path))
        
        assert loaded_model.is_fitted
        assert loaded_model.feature_names == trained_rf_model.feature_names
        assert loaded_model.get_model_name() == trained_rf_model.get_model_name()
    
    def test_predict_with_threshold(self, trained_rf_model, sample_fraud_data):
        """Test prediction with custom threshold."""
        X = sample_fraud_data.drop('Class', axis=1)
        
        # Test with high threshold (should predict fewer frauds)
        pred_high = trained_rf_model.predict_with_threshold(X, threshold=0.9)
        pred_normal = trained_rf_model.predict(X)
        
        assert pred_high.sum() <= pred_normal.sum()


class TestXGBoostFraudModel:
    """Test XGBoostFraudModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = XGBoostFraudModel()
        assert model.get_model_name() == "XGBoost"
    
    def test_fit_predict(self, sample_fraud_data):
        """Test model fitting and prediction."""
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        model = XGBoostFraudModel({'n_estimators': 10, 'random_state': 42})
        model.fit(X, y)
        
        assert model.is_fitted
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)


class TestLightGBMFraudModel:
    """Test LightGBMFraudModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = LightGBMFraudModel()
        assert model.get_model_name() == "LightGBM"
    
    @pytest.mark.slow
    def test_fit_predict(self, sample_fraud_data):
        """Test model fitting and prediction."""
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        model = LightGBMFraudModel({'n_estimators': 10, 'random_state': 42})
        model.fit(X, y)
        
        assert model.is_fitted
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestVotingEnsembleFraudModel:
    """Test VotingEnsembleFraudModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = VotingEnsembleFraudModel()
        assert model.get_model_name() == "VotingEnsemble"
    
    @pytest.mark.slow
    def test_fit_predict(self, small_fraud_data):
        """Test ensemble fitting and prediction."""
        X = small_fraud_data.drop('Class', axis=1)
        y = small_fraud_data['Class']
        
        model = VotingEnsembleFraudModel()
        model.fit(X, y)
        
        assert model.is_fitted
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)


class TestCustomWeightedEnsemble:
    """Test CustomWeightedEnsemble."""
    
    def test_init(self):
        """Test ensemble initialization."""
        ensemble = CustomWeightedEnsemble()
        assert ensemble.get_model_name() == "CustomWeightedEnsemble"
        assert 'rf' in ensemble.models
        assert 'xgb' in ensemble.models
        assert 'lgb' in ensemble.models
    
    @pytest.mark.slow
    def test_fit_predict(self, small_fraud_data):
        """Test ensemble fitting and prediction."""
        X = small_fraud_data.drop('Class', axis=1)
        y = small_fraud_data['Class']
        
        # Split into train/val for weight learning
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        ensemble = CustomWeightedEnsemble({'models': ['rf', 'xgb']})
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        assert ensemble.is_fitted
        assert ensemble.weights is not None
        
        predictions = ensemble.predict(X_val)
        probabilities = ensemble.predict_proba(X_val)
        
        assert len(predictions) == len(X_val)
        assert probabilities.shape == (len(X_val), 2)


class TestMLPFraudModel:
    """Test MLPFraudModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = MLPFraudModel()
        assert model.get_model_name() == "MLP"
    
    def test_fit_predict(self, small_fraud_data):
        """Test MLP fitting and prediction."""
        X = small_fraud_data.drop('Class', axis=1)
        y = small_fraud_data['Class']
        
        model = MLPFraudModel({
            'hidden_layer_sizes': (10, 5),
            'max_iter': 10,
            'random_state': 42
        })
        model.fit(X, y)
        
        assert model.is_fitted
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)


class TestCreateEnsembleModel:
    """Test ensemble model factory function."""
    
    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        model = create_ensemble_model('random_forest')
        assert isinstance(model, RandomForestFraudModel)
    
    def test_create_xgboost(self):
        """Test creating XGBoost model.""" 
        model = create_ensemble_model('xgboost')
        assert isinstance(model, XGBoostFraudModel)
    
    def test_create_voting(self):
        """Test creating Voting ensemble."""
        model = create_ensemble_model('voting')
        assert isinstance(model, VotingEnsembleFraudModel)
    
    def test_create_with_config(self):
        """Test creating model with config."""
        config = {'n_estimators': 50}
        model = create_ensemble_model('random_forest', config)
        
        assert model.config['n_estimators'] == 50
    
    def test_invalid_model_type(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_ensemble_model('invalid_model')


class TestModelRegistry:
    """Test ModelRegistry."""
    
    def test_init(self, tmp_path):
        """Test registry initialization."""
        registry = ModelRegistry(str(tmp_path))
        assert registry.base_path == tmp_path
        assert tmp_path.exists()
    
    def test_register_model(self, trained_rf_model, tmp_path):
        """Test model registration."""
        registry = ModelRegistry(str(tmp_path))
        
        registry.register_model("test_model", trained_rf_model)
        
        assert "test_model" in registry.models
        assert (tmp_path / "test_model.pkl").exists()
    
    def test_get_model(self, trained_rf_model, tmp_path):
        """Test model retrieval."""
        registry = ModelRegistry(str(tmp_path))
        registry.register_model("test_model", trained_rf_model)
        
        retrieved_model = registry.get_model("test_model")
        assert retrieved_model.get_model_name() == trained_rf_model.get_model_name()
    
    def test_list_models(self, trained_rf_model, tmp_path):
        """Test listing models."""
        registry = ModelRegistry(str(tmp_path))
        
        # Initially empty
        models = registry.list_models()
        assert len(models) == 0
        
        # After registration
        registry.register_model("test_model", trained_rf_model)
        models = registry.list_models()
        assert "test_model" in models
    
    def test_compare_models(self, trained_rf_model, trained_xgb_model, 
                           sample_fraud_data, tmp_path):
        """Test model comparison."""
        registry = ModelRegistry(str(tmp_path))
        
        registry.register_model("rf_model", trained_rf_model)
        registry.register_model("xgb_model", trained_xgb_model)
        
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        comparison = registry.compare_models(X, y)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model_name' in comparison.columns
        assert 'roc_auc' in comparison.columns
        assert 'pr_auc' in comparison.columns
    
    def test_get_best_model(self, trained_rf_model, trained_xgb_model,
                           sample_fraud_data, tmp_path):
        """Test getting best model."""
        registry = ModelRegistry(str(tmp_path))
        
        registry.register_model("rf_model", trained_rf_model)
        registry.register_model("xgb_model", trained_xgb_model)
        
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        best_name, best_model = registry.get_best_model(X, y, metric='roc_auc')
        
        assert best_name in ["rf_model", "xgb_model"]
        assert best_model.is_fitted