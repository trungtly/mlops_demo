import pandas as pd
from scipy.stats import ks_2samp
import numpy as np
from typing import Dict, List, Optional

class DataDriftMonitor:
    """
    A class to monitor data drift between a reference dataset and a current dataset.
    """

    def __init__(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        """
        Initialize the data drift monitor.

        Args:
            reference_df (pd.DataFrame): The reference dataset (e.g., training data).
            current_df (pd.DataFrame): The current dataset (e.g., production data).
        """
        self.reference_df = reference_df
        self.current_df = current_df

    def calculate_psi(self, feature: str, num_buckets: int = 10) -> float:
        """
        Calculate the Population Stability Index (PSI) for a single feature.

        Args:
            feature (str): The name of the feature to analyze.
            num_buckets (int): The number of buckets to use for continuous features.

        Returns:
            float: The PSI value.
        """
        try:
            ref_series = self.reference_df[feature]
            current_series = self.current_df[feature]

            # Determine buckets based on the reference data
            if pd.api.types.is_numeric_dtype(ref_series):
                ref_buckets = pd.qcut(ref_series, num_buckets, retbins=True, duplicates='drop')[1]
            else:
                ref_buckets = ref_series.unique()

            ref_dist = ref_series.value_counts(normalize=True)
            current_dist = current_series.value_counts(normalize=True)

            # Align distributions
            dist_df = pd.DataFrame({'ref': ref_dist, 'current': current_dist}).fillna(0)

            # Calculate PSI
            dist_df['psi'] = (dist_df['current'] - dist_df['ref']) * np.log(dist_df['current'] / dist_df['ref'])
            psi_value = dist_df['psi'].sum()

            return psi_value
        except (KeyError, ZeroDivisionError):
            return np.nan


    def run_ks_test(self, feature: str) -> Dict[str, float]:
        """
        Run the Kolmogorov-Smirnov (KS) test for a single feature.

        Args:
            feature (str): The name of the feature to analyze.

        Returns:
            Dict[str, float]: A dictionary containing the KS statistic and p-value.
        """
        try:
            ref_series = self.reference_df[feature]
            current_series = self.current_df[feature]

            ks_stat, p_value = ks_2samp(ref_series, current_series)

            return {'ks_statistic': ks_stat, 'p_value': p_value}
        except KeyError:
            return {'ks_statistic': np.nan, 'p_value': np.nan}

    def get_drift_report(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a drift report for the specified features.

        Args:
            features (Optional[List[str]]): A list of features to check for drift. 
                                            If None, all common features are used.

        Returns:
            pd.DataFrame: A DataFrame containing the drift report.
        """
        if features is None:
            features = list(set(self.reference_df.columns) & set(self.current_df.columns))

        report = []
        for feature in features:
            ks_results = self.run_ks_test(feature)
            psi_value = self.calculate_psi(feature)

            report.append({
                'feature': feature,
                'ks_statistic': ks_results['ks_statistic'],
                'p_value': ks_results['p_value'],
                'psi': psi_value,
                'drift_detected_ks': ks_results['p_value'] < 0.05 if not np.isnan(ks_results['p_value']) else False,
                'drift_severity_psi': self._interpret_psi(psi_value)
            })

        return pd.DataFrame(report)

    @staticmethod
    def _interpret_psi(psi: float) -> str:
        """
        Interpret the PSI value to determine drift severity.

        Args:
            psi (float): The PSI value.

        Returns:
            str: The drift severity ('No drift', 'Minor drift', 'Major drift').
        """
        if np.isnan(psi):
            return "Not applicable"
        if psi < 0.1:
            return 'No drift'
        elif psi < 0.25:
            return 'Minor drift'
        else:
            return 'Major drift'

if __name__ == '__main__':
    # Example usage
    # Create dummy data for demonstration
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.rand(1000) * 10,
        'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
    })

    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.5, 500),  # Drifted
        'feature2': np.random.rand(500) * 10,       # No drift
        'feature3': np.random.choice(['A', 'B', 'C'], 500, p=[0.4, 0.4, 0.2]) # Drifted
    })

    # Initialize the monitor and generate a report
    monitor = DataDriftMonitor(reference_df=reference_data, current_df=current_data)
    drift_report = monitor.get_drift_report()

    print("Data Drift Report:")
    print(drift_report)
