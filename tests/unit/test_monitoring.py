import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from fraud_detection.monitoring.drift import DataDriftMonitor
from fraud_detection.monitoring.performance import PerformanceMonitor

@pytest.fixture(scope="module")
def reference_data():
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.rand(1000) * 10,
        'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
    })

@pytest.fixture(scope="module")
def current_data_no_drift(reference_data):
    return reference_data.sample(500)

@pytest.fixture(scope="module")
def current_data_with_drift():
    return pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.5, 500),
        'feature2': np.random.rand(500) * 10,
        'feature3': np.random.choice(['A', 'B', 'C'], 500, p=[0.4, 0.4, 0.2])
    })

@pytest.fixture(scope="module")
def dummy_model(tmpdir_factory):
    model_path = tmpdir_factory.mktemp("models").join("dummy_model.joblib")
    X, y = pd.DataFrame(abs(np.random.randn(100, 5))), pd.Series(np.random.randint(0, 2, 100))
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return str(model_path)

class TestDataDriftMonitor:
    def test_init(self, reference_data, current_data_no_drift):
        monitor = DataDriftMonitor(reference_data, current_data_no_drift)
        assert monitor.reference_df.equals(reference_data)
        assert monitor.current_df.equals(current_data_no_drift)

    def test_psi_calculation(self, reference_data, current_data_with_drift):
        monitor = DataDriftMonitor(reference_data, current_data_with_drift)
        psi = monitor.calculate_psi('feature1')
        assert isinstance(psi, float)
        assert psi > 0

    def test_ks_test(self, reference_data, current_data_with_drift):
        monitor = DataDriftMonitor(reference_data, current_data_with_drift)
        ks_results = monitor.run_ks_test('feature1')
        assert 'ks_statistic' in ks_results
        assert 'p_value' in ks_results
        assert ks_results['p_value'] < 0.05

    def test_drift_report_no_drift(self, reference_data, current_data_no_drift):
        monitor = DataDriftMonitor(reference_data, current_data_no_drift)
        report = monitor.get_drift_report()
        assert len(report) == 3
        assert not report['drift_detected_ks'].any()

    def test_drift_report_with_drift(self, reference_data, current_data_with_drift):
        monitor = DataDriftMonitor(reference_data, current_data_with_drift)
        report = monitor.get_drift_report()
        assert len(report) == 3
        assert report.loc[report['feature'] == 'feature1', 'drift_detected_ks'].bool()
        assert report.loc[report['feature'] == 'feature3', 'drift_severity_psi'].iloc[0] == 'Major drift'


class TestPerformanceMonitor:
    @pytest.fixture
    def perf_data(self, dummy_model):
        model = joblib.load(dummy_model)
        X_test, y_test = pd.DataFrame(abs(np.random.randn(50, 5))), pd.Series(np.random.randint(0, 2, 50))
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return dummy_model, y_test, y_pred, y_prob

    def test_init(self, perf_data):
        model_path, y_true, y_pred, y_prob = perf_data
        monitor = PerformanceMonitor(model_path, y_true, y_pred, y_prob)
        assert monitor.y_true.equals(y_true)
        assert monitor.y_pred.equals(y_pred)

    def test_calculate_metrics(self, perf_data):
        model_path, y_true, y_pred, y_prob = perf_data
        monitor = PerformanceMonitor(model_path, y_true, y_pred, y_prob)
        metrics = monitor.calculate_metrics()
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics

    def test_compare_with_baseline(self, perf_data):
        model_path, y_true, y_pred, y_prob = perf_data
        monitor = PerformanceMonitor(model_path, y_true, y_pred, y_prob)
        baseline = {
            'accuracy': 0.99,
            'precision': 0.95,
            'recall': 0.90,
            'f1_score': 0.92,
            'roc_auc': 0.98,
            'pr_auc': 0.97
        }
        comparison = monitor.compare_with_baseline(baseline)
        assert 'accuracy' in comparison
        assert comparison['accuracy']['status'] in ['OK', 'Warning', 'Critical']
