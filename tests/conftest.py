import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fraud_detection.data.ingestion import FraudDataIngestion
from fraud_detection.models import RandomForestFraudModel, XGBoostFraudModel


@pytest.fixture
def sample_fraud_data():
    """Generate sample fraud detection data for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 28
    
    # Generate V1-V28 features (simulate PCA features)
    V_features = np.random.randn(n_samples, n_features)
    
    # Time feature (in seconds)
    time_feature = np.random.randint(0, 172800, n_samples)  # 2 days in seconds
    
    # Amount feature (log-normal distribution)
    amount_feature = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)
    
    # Create fraud labels (imbalanced - 2% fraud)
    fraud_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    labels = np.zeros(n_samples)
    labels[fraud_indices] = 1
    
    # Make fraud cases more extreme in some V features
    for idx in fraud_indices:
        V_features[idx, :5] += np.random.randn(5) * 2  # Make more extreme
        amount_feature[idx] *= np.random.uniform(0.1, 5.0)  # Vary amounts
    
    # Create DataFrame
    data = pd.DataFrame(V_features, columns=[f'V{i}' for i in range(1, n_features + 1)])
    data['Time'] = time_feature
    data['Amount'] = amount_feature
    data['Class'] = labels.astype(int)
    
    return data


@pytest.fixture
def small_fraud_data():
    """Generate small dataset for quick tests."""
    np.random.seed(42)
    
    n_samples = 100
    
    # Simple 5-feature dataset
    data = pd.DataFrame({
        'V1': np.random.randn(n_samples),
        'V2': np.random.randn(n_samples),
        'V3': np.random.randn(n_samples),
        'Time': np.random.randint(0, 86400, n_samples),
        'Amount': np.random.lognormal(2, 1, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    return data


@pytest.fixture
def trained_rf_model(sample_fraud_data):
    """Fixture providing a trained Random Forest model."""
    X = sample_fraud_data.drop('Class', axis=1)
    y = sample_fraud_data['Class']
    
    model = RandomForestFraudModel({
        'n_estimators': 10,  # Small for speed
        'max_depth': 5,
        'random_state': 42
    })
    
    model.fit(X, y)
    return model


@pytest.fixture
def trained_xgb_model(sample_fraud_data):
    """Fixture providing a trained XGBoost model."""
    X = sample_fraud_data.drop('Class', axis=1)
    y = sample_fraud_data['Class']
    
    model = XGBoostFraudModel({
        'n_estimators': 10,  # Small for speed
        'max_depth': 3,
        'random_state': 42
    })
    
    model.fit(X, y)
    return model


@pytest.fixture
def feature_names():
    """Standard feature names for credit card fraud dataset."""
    return ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]


@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary path for saving/loading models."""
    return tmp_path / "test_model.pkl"


@pytest.fixture
def mock_kaggle_data():
    """Mock Kaggle API response for testing data ingestion."""
    # This would be used with pytest-mock to mock kagglehub responses
    return {
        'dataset_path': '/mock/path/creditcard.csv',
        'files': ['creditcard.csv']
    }


@pytest.fixture
def config_data():
    """Sample configuration data."""
    return {
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42
        },
        'features': {
            'engineering': {
                'include_time': True,
                'include_amount': True,
                'include_statistical': True
            },
            'selection': {
                'method': 'ensemble',
                'n_features': 30
            }
        }
    }


@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory):
    """MLflow tracking URI for testing."""
    tracking_dir = tmp_path_factory.mktemp("mlflow")
    return f"file://{tracking_dir}"


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )


# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("-m"):
        return  # Don't modify if markers are explicitly specified
    
    skip_slow = pytest.mark.skip(reason="slow test skipped (use -m slow to run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)# TODO: add parameterized fixtures for edge cases

# Shared fixtures for fraud detection test suite
