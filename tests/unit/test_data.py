import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from fraud_detection.data.ingestion import FraudDataIngestion
from fraud_detection.data.preprocessing import FraudDataPreprocessor
from fraud_detection.data.validation import FraudDataValidator


class TestFraudDataIngestion:
    """Test FraudDataIngestion class."""
    
    def test_init(self):
        """Test initialization."""
        ingestion = FraudDataIngestion()
        assert ingestion is not None
    
    def test_load_local_data(self, sample_fraud_data, tmp_path):
        """Test loading local CSV data."""
        # Save sample data to temporary file
        test_file = tmp_path / "test_data.csv"
        sample_fraud_data.to_csv(test_file, index=False)
        
        ingestion = FraudDataIngestion()
        loaded_data = ingestion.load_local_data(str(test_file))
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_fraud_data.shape
        assert list(loaded_data.columns) == list(sample_fraud_data.columns)
        assert loaded_data['Class'].sum() == sample_fraud_data['Class'].sum()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        ingestion = FraudDataIngestion()
        
        with pytest.raises(FileNotFoundError):
            ingestion.load_local_data("nonexistent_file.csv")
    
    @patch('fraud_detection.data.ingestion.kagglehub.dataset_download')
    def test_download_creditcard_data(self, mock_download, sample_fraud_data, tmp_path):
        """Test downloading credit card data from Kaggle."""
        # Mock kagglehub response
        test_file = tmp_path / "creditcard.csv"
        sample_fraud_data.to_csv(test_file, index=False)
        mock_download.return_value = str(tmp_path)
        
        ingestion = FraudDataIngestion()
        result = ingestion.download_creditcard_data()
        
        assert isinstance(result, pd.DataFrame)
        mock_download.assert_called_once_with("mlg-ulb/creditcardfraud")
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        ingestion = FraudDataIngestion()
        
        n_samples = 1000
        fraud_rate = 0.02
        
        data = ingestion.generate_synthetic_data(
            n_samples=n_samples, 
            fraud_rate=fraud_rate
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == n_samples
        assert 'Class' in data.columns
        assert 'Time' in data.columns
        assert 'Amount' in data.columns
        
        # Check fraud rate is approximately correct
        actual_fraud_rate = data['Class'].mean()
        assert abs(actual_fraud_rate - fraud_rate) < 0.01
        
        # Check V features exist
        v_columns = [col for col in data.columns if col.startswith('V')]
        assert len(v_columns) >= 10  # Should have some V features


class TestFraudDataPreprocessor:
    """Test FraudDataPreprocessor class."""
    
    def test_init(self):
        """Test initialization."""
        preprocessor = FraudDataPreprocessor()
        assert preprocessor is not None
    
    def test_preprocess_for_training(self, sample_fraud_data):
        """Test preprocessing for training."""
        preprocessor = FraudDataPreprocessor()
        
        result = preprocessor.preprocess_for_training(
            sample_fraud_data, 
            test_size=0.2, 
            validation_size=0.1,
            random_state=42
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = result
        
        # Check shapes
        total_samples = len(sample_fraud_data)
        expected_test = int(total_samples * 0.2)
        expected_val = int((total_samples - expected_test) * 0.1)
        expected_train = total_samples - expected_test - expected_val
        
        assert len(X_train) == expected_train
        assert len(X_val) == expected_val  
        assert len(X_test) == expected_test
        
        # Check that Class column is removed from features
        assert 'Class' not in X_train.columns
        assert 'Class' not in X_val.columns
        assert 'Class' not in X_test.columns
        
        # Check targets
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)
    
    def test_handle_missing_values(self, sample_fraud_data):
        """Test missing value handling."""
        # Introduce missing values
        data_with_missing = sample_fraud_data.copy()
        data_with_missing.loc[0:10, 'V1'] = np.nan
        data_with_missing.loc[5:15, 'Amount'] = np.nan
        
        preprocessor = FraudDataPreprocessor()
        result = preprocessor.handle_missing_values(data_with_missing)
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
        assert len(result) == len(data_with_missing)
    
    def test_remove_duplicates(self, sample_fraud_data):
        """Test duplicate removal."""
        # Add duplicates
        data_with_dups = pd.concat([sample_fraud_data, sample_fraud_data.iloc[:10]], 
                                  ignore_index=True)
        
        preprocessor = FraudDataPreprocessor()
        result = preprocessor.remove_duplicates(data_with_dups)
        
        # Should have no duplicates
        assert len(result) == len(sample_fraud_data)
    
    def test_handle_outliers(self, sample_fraud_data):
        """Test outlier handling."""
        preprocessor = FraudDataPreprocessor()
        
        # Test IQR method
        result_iqr = preprocessor.handle_outliers(
            sample_fraud_data, 
            method='iqr',
            columns=['Amount']
        )
        
        assert isinstance(result_iqr, pd.DataFrame)
        assert len(result_iqr) <= len(sample_fraud_data)
        
        # Test Z-score method
        result_zscore = preprocessor.handle_outliers(
            sample_fraud_data,
            method='zscore', 
            columns=['Amount']
        )
        
        assert isinstance(result_zscore, pd.DataFrame)
        assert len(result_zscore) <= len(sample_fraud_data)


class TestFraudDataValidator:
    """Test FraudDataValidator class."""
    
    def test_init(self):
        """Test initialization."""
        validator = FraudDataValidator()
        assert validator is not None
    
    def test_validate_schema(self, sample_fraud_data):
        """Test schema validation."""
        validator = FraudDataValidator()
        
        # Valid schema
        is_valid, errors = validator.validate_schema(sample_fraud_data)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid schema - missing Class column
        invalid_data = sample_fraud_data.drop('Class', axis=1)
        is_valid, errors = validator.validate_schema(invalid_data)
        assert not is_valid
        assert len(errors) > 0
    
    def test_check_data_quality(self, sample_fraud_data):
        """Test data quality checks."""
        validator = FraudDataValidator()
        
        quality_report = validator.check_data_quality(sample_fraud_data)
        
        assert 'missing_values' in quality_report
        assert 'duplicates' in quality_report
        assert 'outliers' in quality_report
        assert 'class_distribution' in quality_report
        
        # Check class distribution
        fraud_rate = quality_report['class_distribution']['fraud_rate']
        assert 0 <= fraud_rate <= 1
    
    def test_detect_data_drift(self, sample_fraud_data):
        """Test data drift detection."""
        validator = FraudDataValidator()
        
        # Create slightly modified reference data
        reference_data = sample_fraud_data.copy()
        reference_data['V1'] += np.random.normal(0, 0.1, len(reference_data))
        
        drift_report = validator.detect_data_drift(
            sample_fraud_data, 
            reference_data
        )
        
        assert 'drift_detected' in drift_report
        assert 'feature_drifts' in drift_report
        assert isinstance(drift_report['drift_detected'], bool)
    
    def test_validate_fraud_rates(self, sample_fraud_data):
        """Test fraud rate validation."""
        validator = FraudDataValidator()
        
        # Normal fraud rate
        is_valid, message = validator.validate_fraud_rates(sample_fraud_data)
        assert is_valid
        
        # Abnormal fraud rate (too high)
        high_fraud_data = sample_fraud_data.copy()
        high_fraud_data['Class'] = 1  # All fraud
        
        is_valid, message = validator.validate_fraud_rates(high_fraud_data)
        assert not is_valid
        assert "unusually high" in message.lower()
    
    def test_validate_feature_distributions(self, sample_fraud_data):
        """Test feature distribution validation."""
        validator = FraudDataValidator()
        
        validation_results = validator.validate_feature_distributions(sample_fraud_data)
        
        assert isinstance(validation_results, dict)
        assert 'valid_features' in validation_results
        assert 'invalid_features' in validation_results
        
        # Amount should be positive
        assert 'Amount' in validation_results['valid_features'] or \
               'Amount' in validation_results['invalid_features']