import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from fraud_detection.data.ingestion import FraudDataIngestion
from fraud_detection.data.preprocessing import FraudDataPreprocessor
from fraud_detection.features.engineering import FraudFeatureEngineer
from fraud_detection.features.selection import FraudFeatureSelector
from fraud_detection.models import RandomForestFraudModel, XGBoostFraudModel
from fraud_detection.models.base import ModelRegistry
from fraud_detection.evaluation.metrics import FraudMetrics
from fraud_detection.training.train import FraudTrainer


class TestEndToEndPipeline:
    """Test complete end-to-end fraud detection pipeline."""
    
    @pytest.mark.integration
    def test_complete_pipeline(self, sample_fraud_data, tmp_path):
        """Test complete pipeline from data to trained model."""
        
        # 1. Data Ingestion - Save and load data
        data_file = tmp_path / "test_data.csv"
        sample_fraud_data.to_csv(data_file, index=False)
        
        ingestion = FraudDataIngestion()
        df = ingestion.load_local_data(str(data_file))
        
        assert len(df) == len(sample_fraud_data)
        
        # 2. Data Preprocessing
        preprocessor = FraudDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
            df, test_size=0.2, validation_size=0.1, random_state=42
        )
        
        assert len(X_train) + len(X_val) + len(X_test) == len(df)
        
        # 3. Feature Engineering
        feature_engineer = FraudFeatureEngineer()
        X_train_eng = feature_engineer.engineer_features(
            X_train, include_rolling=False, include_interactions=False
        )
        X_val_eng = feature_engineer.engineer_features(
            X_val, include_rolling=False, include_interactions=False
        )
        X_test_eng = feature_engineer.engineer_features(
            X_test, include_rolling=False, include_interactions=False
        )
        
        assert X_train_eng.shape[1] > X_train.shape[1]  # More features after engineering
        
        # 4. Feature Selection
        feature_selector = FraudFeatureSelector()
        X_train_selected = feature_selector.select_features(
            X_train_eng, y_train, method='univariate', k=20
        )
        
        selected_features = X_train_selected.columns.tolist()
        X_val_selected = X_val_eng[selected_features]
        X_test_selected = X_test_eng[selected_features]
        
        assert X_train_selected.shape[1] <= X_train_eng.shape[1]  # Fewer features after selection
        
        # 5. Model Training
        model = RandomForestFraudModel({
            'n_estimators': 10, 
            'max_depth': 5, 
            'random_state': 42
        })
        
        model.fit(X_train_selected, y_train, X_val_selected, y_val)
        assert model.is_fitted
        
        # 6. Model Evaluation
        metrics_calculator = FraudMetrics()
        
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        test_metrics = metrics_calculator.calculate_metrics(y_test, y_pred_proba)
        
        assert 'roc_auc' in test_metrics
        assert 'pr_auc' in test_metrics
        assert 0 <= test_metrics['roc_auc'] <= 1
        assert 0 <= test_metrics['pr_auc'] <= 1
        
        # 7. Model Registry
        registry = ModelRegistry(str(tmp_path / "models"))
        registry.register_model("test_model", model)
        
        retrieved_model = registry.get_model("test_model")
        assert retrieved_model.get_model_name() == model.get_model_name()
    
    @pytest.mark.integration
    def test_multiple_model_comparison(self, sample_fraud_data, tmp_path):
        """Test training and comparing multiple models."""
        
        # Prepare data
        preprocessor = FraudDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
            sample_fraud_data, test_size=0.2, random_state=42
        )
        
        # Simple feature engineering
        feature_engineer = FraudFeatureEngineer()
        X_train_eng = feature_engineer.engineer_features(
            X_train, include_rolling=False, include_interactions=False, include_outliers=False
        )
        X_val_eng = feature_engineer.engineer_features(
            X_val, include_rolling=False, include_interactions=False, include_outliers=False
        )
        X_test_eng = feature_engineer.engineer_features(
            X_test, include_rolling=False, include_interactions=False, include_outliers=False
        )
        
        # Train multiple models
        models = {
            'rf': RandomForestFraudModel({
                'n_estimators': 10, 
                'max_depth': 5, 
                'random_state': 42
            }),
            'xgb': XGBoostFraudModel({
                'n_estimators': 10, 
                'max_depth': 3, 
                'random_state': 42
            })
        }
        
        registry = ModelRegistry(str(tmp_path / "models"))
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_eng, y_train, X_val_eng, y_val)
            
            # Evaluate model
            test_metrics = model.evaluate(X_test_eng, y_test)
            results[name] = test_metrics
            
            # Register model
            registry.register_model(name, model)
        
        # Compare models
        comparison = registry.compare_models(X_test_eng, y_test)
        
        assert len(comparison) == len(models)
        assert all(model_name in comparison['model_name'].values for model_name in models.keys())
        
        # Get best model
        best_name, best_model = registry.get_best_model(X_test_eng, y_test, metric='pr_auc')
        assert best_name in models.keys()
        assert best_model.is_fitted
    
    @pytest.mark.integration 
    @pytest.mark.slow
    def test_trainer_integration(self, sample_fraud_data):
        """Test FraudTrainer integration."""
        
        # Prepare data
        preprocessor = FraudDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
            sample_fraud_data, test_size=0.2, random_state=42
        )
        
        # Create model
        model = RandomForestFraudModel({
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42
        })
        
        # Initialize trainer
        trainer = FraudTrainer(
            enable_hyperparameter_tuning=False,  # Disable for speed
            cv_folds=3,
            random_state=42
        )
        
        # Train model
        trained_model = trainer.train_model(model, X_train, y_train, X_val, y_val)
        
        assert trained_model.is_fitted
        
        # Test predictions
        predictions = trained_model.predict_proba(X_test)[:, 1]
        assert len(predictions) == len(X_test)
        assert all(0 <= p <= 1 for p in predictions)
    
    @pytest.mark.integration
    def test_synthetic_data_pipeline(self, tmp_path):
        """Test pipeline with synthetic data."""
        
        # Generate synthetic data
        ingestion = FraudDataIngestion()
        synthetic_data = ingestion.generate_synthetic_data(
            n_samples=1000, 
            fraud_rate=0.02
        )
        
        # Run through preprocessing
        preprocessor = FraudDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
            synthetic_data, test_size=0.2, random_state=42
        )
        
        # Feature engineering
        feature_engineer = FraudFeatureEngineer()
        X_train_eng = feature_engineer.create_baseline_features(X_train)
        X_val_eng = feature_engineer.create_baseline_features(X_val)
        X_test_eng = feature_engineer.create_baseline_features(X_test)
        
        # Train simple model
        model = RandomForestFraudModel({
            'n_estimators': 10,
            'random_state': 42
        })
        
        model.fit(X_train_eng, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test_eng, y_test)
        
        # Should get reasonable performance on synthetic data
        assert metrics['roc_auc'] > 0.5  # Better than random
        assert metrics['pr_auc'] > 0.01  # Better than baseline fraud rate
    
    @pytest.mark.integration
    def test_feature_pipeline_consistency(self, sample_fraud_data):
        """Test that feature engineering is consistent across train/val/test."""
        
        # Split data
        preprocessor = FraudDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
            sample_fraud_data, test_size=0.2, random_state=42
        )
        
        # Apply feature engineering
        feature_engineer = FraudFeatureEngineer()
        
        X_train_eng = feature_engineer.engineer_features(X_train, scale=False)  # Don't scale yet
        X_val_eng = feature_engineer.engineer_features(X_val, scale=False)
        X_test_eng = feature_engineer.engineer_features(X_test, scale=False)
        
        # Check that all sets have same columns
        assert list(X_train_eng.columns) == list(X_val_eng.columns)
        assert list(X_val_eng.columns) == list(X_test_eng.columns)
        
        # Check that engineered features exist
        engineered_features = [
            'Hour_sin', 'Hour_cos', 'Amount_log', 'V_mean', 'V_std'
        ]
        
        for feature in engineered_features:
            assert feature in X_train_eng.columns
            assert feature in X_val_eng.columns
            assert feature in X_test_eng.columns
    
    @pytest.mark.integration
    def test_model_persistence(self, sample_fraud_data, tmp_path):
        """Test model saving/loading pipeline."""
        
        # Train model
        X = sample_fraud_data.drop('Class', axis=1)
        y = sample_fraud_data['Class']
        
        model = RandomForestFraudModel({
            'n_estimators': 10,
            'random_state': 42
        })
        model.fit(X, y)
        
        # Get predictions before saving
        predictions_before = model.predict_proba(X)[:, 1]
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save_model(str(model_path), save_mlflow=False)
        
        # Load model
        new_model = RandomForestFraudModel()
        loaded_model = new_model.load_model(str(model_path))
        
        # Get predictions after loading
        predictions_after = loaded_model.predict_proba(X)[:, 1]
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(predictions_before, predictions_after)
        
        # Model metadata should be preserved
        assert loaded_model.get_model_name() == model.get_model_name()
        assert loaded_model.feature_names == model.feature_names
    
    @pytest.mark.integration
    def test_error_handling_pipeline(self, tmp_path):
        """Test pipeline error handling."""
        
        # Test with missing required columns
        invalid_data = pd.DataFrame({
            'V1': [1, 2, 3],
            'V2': [4, 5, 6]
            # Missing 'Class' column
        })
        
        preprocessor = FraudDataPreprocessor()
        
        with pytest.raises((KeyError, ValueError)):
            preprocessor.preprocess_for_training(invalid_data)
        
        # Test with empty dataset
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            preprocessor.preprocess_for_training(empty_data)
    
    @pytest.mark.integration
    def test_configuration_driven_pipeline(self, sample_fraud_data, config_data, tmp_path):
        """Test pipeline driven by configuration."""
        
        # Use config to drive pipeline choices
        model_config = config_data['models']['random_forest']
        training_config = config_data['training']
        
        # Data preprocessing with config
        preprocessor = FraudDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(
            sample_fraud_data, 
            test_size=training_config['test_size'],
            random_state=training_config['random_state']
        )
        
        # Model creation with config
        model = RandomForestFraudModel(model_config)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Save configuration and results
        results = {
            'config': config_data,
            'metrics': metrics,
            'model_info': model.get_model_info()
        }
        
        results_file = tmp_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        assert results_file.exists()
        
        # Load and verify results
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'config' in loaded_results
        assert 'metrics' in loaded_results
        assert 'model_info' in loaded_results