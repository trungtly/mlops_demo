"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from ..config import config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for fraud detection."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Initialize feature engineering pipeline.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or {}
        self.scalers = {}
        self.feature_stats = {}
        self.pca_components = None
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the Time column.
        
        Args:
            df: Input DataFrame with Time column
            
        Returns:
            DataFrame with time features
        """
        if 'Time' not in df.columns:
            logger.warning("Time column not found. Skipping time features.")
            return df.copy()
            
        logger.info("Creating time-based features")
        df_copy = df.copy()
        
        # Convert seconds to hours (assuming Time is in seconds)
        df_copy['Hour'] = (df_copy['Time'] / 3600) % 24
        
        # Cyclical encoding of hour
        df_copy['Hour_sin'] = np.sin(2 * np.pi * df_copy['Hour'] / 24)
        df_copy['Hour_cos'] = np.cos(2 * np.pi * df_copy['Hour'] / 24)
        
        # Day of transaction (assuming consecutive days)
        df_copy['Day'] = (df_copy['Time'] / (24 * 3600)).astype(int)
        
        # Time since first transaction
        df_copy['Time_normalized'] = (df_copy['Time'] - df_copy['Time'].min()) / (df_copy['Time'].max() - df_copy['Time'].min() + 1e-8)
        
        # Time of day segments
        df_copy['is_morning'] = ((df_copy['Hour'] >= 5) & (df_copy['Hour'] < 12)).astype(int)
        df_copy['is_afternoon'] = ((df_copy['Hour'] >= 12) & (df_copy['Hour'] < 17)).astype(int)
        df_copy['is_evening'] = ((df_copy['Hour'] >= 17) & (df_copy['Hour'] < 22)).astype(int)
        df_copy['is_night'] = ((df_copy['Hour'] >= 22) | (df_copy['Hour'] < 5)).astype(int)
        
        # Weekend detection (approximate since we don't have real dates)
        df_copy['is_weekend'] = (df_copy['Day'] % 7 >= 5).astype(int)
        
        return df_copy
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features.
        
        Args:
            df: Input DataFrame with Amount column
            
        Returns:
            DataFrame with amount features
        """
        if 'Amount' not in df.columns:
            logger.warning("Amount column not found. Skipping amount features.")
            return df.copy()
            
        logger.info("Creating amount-based features")
        df_copy = df.copy()
        
        # Log transformation for amount (add small constant to avoid log(0))
        df_copy['Amount_log'] = np.log1p(df_copy['Amount'])
        
        # Amount quantiles
        df_copy['Amount_quantile'] = pd.qcut(
            df_copy['Amount'], 
            q=10, 
            labels=False, 
            duplicates='drop'
        )
        
        # Amount categories
        amount_thresholds = [0, 10, 50, 100, 500, 1000, np.inf]
        df_copy['Amount_category'] = pd.cut(
            df_copy['Amount'], 
            bins=amount_thresholds, 
            labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme']
        )
        
        # Square root transformation
        df_copy['Amount_sqrt'] = np.sqrt(df_copy['Amount'])
        
        # Round number detection (potential anomalies)
        df_copy['is_round_amount_10'] = ((df_copy['Amount'] % 10) < 0.01).astype(int)
        df_copy['is_round_amount_100'] = ((df_copy['Amount'] % 100) < 0.01).astype(int)
        
        # Unusually high amount flag (above 95th percentile)
        high_amount_threshold = df_copy['Amount'].quantile(0.95)
        df_copy['is_high_amount'] = (df_copy['Amount'] > high_amount_threshold).astype(int)
        
        return df_copy
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from existing V columns.
        
        Args:
            df: Input DataFrame with V columns
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("Creating statistical features")
        df_copy = df.copy()
        
        # V columns (PCA features)
        v_columns = [col for col in df.columns if col.startswith('V')]
        
        if not v_columns:
            logger.warning("No V columns found. Skipping statistical features.")
            return df_copy
            
        # Statistical aggregations
        df_copy['V_mean'] = df_copy[v_columns].mean(axis=1)
        df_copy['V_std'] = df_copy[v_columns].std(axis=1)
        df_copy['V_min'] = df_copy[v_columns].min(axis=1)
        df_copy['V_max'] = df_copy[v_columns].max(axis=1)
        df_copy['V_range'] = df_copy['V_max'] - df_copy['V_min']
        df_copy['V_median'] = df_copy[v_columns].median(axis=1)
        df_copy['V_skew'] = df_copy[v_columns].skew(axis=1)
        df_copy['V_kurtosis'] = df_copy[v_columns].kurtosis(axis=1)
        
        # Number of extreme values (beyond 2 standard deviations)
        v_array = df_copy[v_columns].values
        df_copy['V_extreme_count'] = np.sum(np.abs(v_array) > 2, axis=1)
        
        # Entropy-like measure
        v_abs = np.abs(v_array)
        v_sum = v_abs.sum(axis=1)
        v_normalized = v_abs / (v_sum.reshape(-1, 1) + 1e-8)
        df_copy['V_entropy'] = -np.sum(
            v_normalized * np.log(v_normalized + 1e-8), axis=1
        )
        
        # Higher order statistics
        df_copy['V_mean_abs_diff'] = np.mean(
            np.abs(v_array - df_copy['V_mean'].values.reshape(-1, 1)), axis=1
        )
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        df_copy = df.copy()
        
        # Amount-Time interactions
        if 'Amount' in df_copy.columns and 'Time_normalized' in df_copy.columns:
            df_copy['Amount_Time_ratio'] = df_copy['Amount'] / (df_copy['Time_normalized'] + 1e-8)
        
        # Amount by hour patterns
        if 'Amount' in df_copy.columns and 'Hour' in df_copy.columns:
            df_copy['Amount_Hour_interaction'] = df_copy['Amount'] * df_copy['Hour']
        
        # High amount at unusual times (night hours)
        if 'Amount' in df_copy.columns and 'is_night' in df_copy.columns:
            high_amount = (df_copy['Amount'] > df_copy['Amount'].quantile(0.9))
            df_copy['High_amount_at_night'] = (high_amount & (df_copy['is_night'] == 1)).astype(int)
        
        # Amount to V feature interactions
        v_columns = [col for col in df.columns if col.startswith('V')]
        if 'Amount' in df_copy.columns and v_columns:
            # Get top 5 V features with highest variance
            v_var = df_copy[v_columns].var().sort_values(ascending=False)
            top_v_cols = v_var.index[:5].tolist()
            
            for v_col in top_v_cols:
                df_copy[f'{v_col}_Amount_ratio'] = df_copy[v_col] / (df_copy['Amount'] + 1e-8)
                df_copy[f'{v_col}_Amount_product'] = df_copy[v_col] * df_copy['Amount']
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [10, 50, 100]) -> pd.DataFrame:
        """
        Create rolling window features (requires sorted data by time).
        
        Args:
            df: Input DataFrame
            window_sizes: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling window features")
        
        if 'Time' not in df.columns:
            logger.warning("Time column not found. Skipping rolling features.")
            return df.copy()
            
        # Sort by time
        df_copy = df.copy().sort_values('Time')
        
        for window in window_sizes:
            # Rolling mean and std of amount
            if 'Amount' in df_copy.columns:
                df_copy[f'Amount_rolling_mean_{window}'] = (
                    df_copy['Amount'].rolling(window=window, min_periods=1).mean()
                )
                df_copy[f'Amount_rolling_std_{window}'] = (
                    df_copy['Amount'].rolling(window=window, min_periods=1).std()
                )
            
            # Transaction velocity (count of transactions per window)
            df_copy[f'Transaction_velocity_{window}'] = (
                df_copy.index.to_series().rolling(window=window, min_periods=1).count()
            )
            
            # Rolling fraud rate (if we have the target variable)
            if 'Class' in df_copy.columns:
                df_copy[f'Fraud_rate_rolling_{window}'] = (
                    df_copy['Class'].rolling(window=window, min_periods=1).mean()
                )
        
        # Resort by index to maintain original order
        df_copy = df_copy.sort_index()
        
        return df_copy
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Add outlier detection features.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier features
        """
        logger.info("Creating outlier detection features")
        df_copy = df.copy()
        
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols 
                         if col not in ['Class', 'Time'] and not col.endswith('_outlier')]
        
        if not numerical_cols:
            logger.warning("No numerical columns found. Skipping outlier detection.")
            return df_copy
            
        outlier_flags = pd.DataFrame(index=df_copy.index)
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_flags[f'{col}_outlier'] = (
                    (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
                ).astype(int)
        
        elif method == 'zscore':
            for col in numerical_cols:
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                if std > 0:  # Avoid division by zero
                    z_scores = (df_copy[col] - mean) / std
                    outlier_flags[f'{col}_outlier'] = (np.abs(z_scores) > threshold).astype(int)
        
        # Add outlier columns to result
        for col in outlier_flags.columns:
            df_copy[col] = outlier_flags[col]
        
        # Count total outliers per row
        outlier_cols = outlier_flags.columns
        df_copy['Total_outliers'] = df_copy[outlier_cols].sum(axis=1)
        df_copy['Outlier_ratio'] = df_copy['Total_outliers'] / len(outlier_cols)
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, is_training: bool = True, scaler_type: str = 'robust') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data
            scaler_type: Type of scaler ('standard', 'robust', or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {scaler_type} scaler")
        df_copy = df.copy()
        
        # Select columns to scale (exclude categorical and special columns)
        skip_cols = ['Class', 'Time']
        categorical_cols = df_copy.select_dtypes(include=['category', 'object']).columns
        skip_cols.extend(categorical_cols)
        skip_cols.extend([col for col in df_copy.columns if col.startswith('is_')])
        
        scale_columns = [col for col in df_copy.columns 
                       if col not in skip_cols and pd.api.types.is_numeric_dtype(df_copy[col])]
        
        if not scale_columns:
            logger.warning("No numerical columns found for scaling.")
            return df_copy
            
        # Create scaler based on type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler(quantile_range=(1.0, 99.0))
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Using RobustScaler.")
            scaler = RobustScaler(quantile_range=(1.0, 99.0))
        
        # Fit scaler on training data or use previously fitted scaler
        if is_training:
            df_copy[scale_columns] = scaler.fit_transform(df_copy[scale_columns])
            self.scalers[scaler_type] = scaler
        else:
            if scaler_type not in self.scalers:
                logger.warning(f"No {scaler_type} scaler found from training data. Fitting on this data.")
                df_copy[scale_columns] = scaler.fit_transform(df_copy[scale_columns])
                self.scalers[scaler_type] = scaler
            else:
                df_copy[scale_columns] = self.scalers[scaler_type].transform(df_copy[scale_columns])
        
        return df_copy
    
    def create_pca_features(self, df: pd.DataFrame, is_training: bool = True, n_components: int = 5) -> pd.DataFrame:
        """
        Create PCA features from V columns.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data
            n_components: Number of PCA components
            
        Returns:
            DataFrame with PCA features
        """
        logger.info(f"Creating {n_components} PCA features")
        df_copy = df.copy()
        
        # Get V columns
        v_columns = [col for col in df.columns if col.startswith('V')]
        
        if not v_columns:
            logger.warning("No V columns found. Skipping PCA features.")
            return df_copy
            
        # Create PCA on training data or use previously fitted PCA
        if is_training or self.pca_components is None:
            pca = PCA(n_components=min(n_components, len(v_columns)))
            pca_result = pca.fit_transform(df_copy[v_columns])
            self.pca_components = pca
            
            # Log explained variance
            total_var = np.sum(pca.explained_variance_ratio_)
            logger.info(f"PCA with {n_components} components explains {total_var:.2%} of variance")
            
        else:
            pca_result = self.pca_components.transform(df_copy[v_columns])
        
        # Add PCA components to DataFrame
        for i in range(pca_result.shape[1]):
            df_copy[f'PCA_{i+1}'] = pca_result[:, i]
        
        return df_copy
    
    def create_features(self, 
                       df: pd.DataFrame, 
                       is_training: bool = True,
                       create_time_features: bool = True,
                       create_amount_features: bool = True,
                       create_statistical: bool = True,
                       create_interactions: bool = True,
                       create_pca: bool = False,
                       create_rolling: bool = False,
                       detect_outliers: bool = True,
                       scale_features: bool = True,
                       scaler_type: str = 'robust') -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data
            create_time_features: Whether to create time-based features
            create_amount_features: Whether to create amount-based features
            create_statistical: Whether to create statistical features
            create_interactions: Whether to create interaction features
            create_pca: Whether to create PCA features
            create_rolling: Whether to create rolling window features
            detect_outliers: Whether to add outlier detection features
            scale_features: Whether to scale features
            scaler_type: Type of scaler ('standard', 'robust', or 'minmax')
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering on data with shape {df.shape}")
        df_engineered = df.copy()
        
        # Apply feature transformations
        if create_time_features:
            df_engineered = self.create_time_features(df_engineered)
            
        if create_amount_features:
            df_engineered = self.create_amount_features(df_engineered)
            
        if create_statistical:
            df_engineered = self.create_statistical_features(df_engineered)
            
        if create_interactions:
            df_engineered = self.create_interaction_features(df_engineered)
            
        if create_rolling:
            df_engineered = self.create_rolling_features(df_engineered)
            
        if detect_outliers:
            df_engineered = self.detect_outliers(df_engineered)
            
        if create_pca:
            df_engineered = self.create_pca_features(df_engineered, is_training)
            
        if scale_features:
            df_engineered = self.scale_features(df_engineered, is_training, scaler_type)
            
        # Log feature stats
        feature_count = df_engineered.shape[1] - df.shape[1]
        logger.info(f"Feature engineering complete. Added {feature_count} new features.")
        logger.info(f"Final data shape: {df_engineered.shape}")
        
        return df_engineered
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get statistics for each feature in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'null_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique()
                }
                
                if 'Class' in df.columns:
                    class_0_mean = df.loc[df['Class'] == 0, col].mean()
                    class_1_mean = df.loc[df['Class'] == 1, col].mean()
                    mean_diff = abs(class_1_mean - class_0_mean)
                    
                    stats[col]['class_0_mean'] = class_0_mean
                    stats[col]['class_1_mean'] = class_1_mean
                    stats[col]['class_mean_diff'] = mean_diff
                    stats[col]['class_mean_diff_ratio'] = mean_diff / (abs(class_0_mean) + 1e-8)
        
        self.feature_stats = stats
        return stats


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick baseline feature engineering for rapid prototyping.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with baseline features
    """
    engineer = FeatureEngineer()
    
    return engineer.create_features(
        df,
        create_time_features=True,
        create_amount_features=True,
        create_statistical=True,
        create_interactions=False,
        create_pca=False,
        create_rolling=False,
        detect_outliers=True,
        scale_features=True
    )


def create_advanced_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Advanced feature engineering with all features.
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data
        
    Returns:
        DataFrame with advanced features
    """
    engineer = FeatureEngineer()
    
    return engineer.create_features(
        df,
        is_training=is_training,
        create_time_features=True,
        create_amount_features=True,
        create_statistical=True,
        create_interactions=True,
        create_pca=True,
        create_rolling=True,
        detect_outliers=True,
        scale_features=True
    )


def main():
    """Run feature engineering as a standalone module."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path if run as standalone script
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    from src.fraud_detection.data.ingestion import FraudDataIngestion
    
    logger.info("Loading data")
    data_ingestion = FraudDataIngestion()
    df, _ = data_ingestion.run_full_ingestion()
    
    # Create features
    logger.info("Creating baseline features")
    baseline_features = create_baseline_features(df)
    
    logger.info("Creating advanced features")
    advanced_features = create_advanced_features(df)
    
    # Print results
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 50)
    print(f"Original data shape: {df.shape}")
    print(f"Baseline features shape: {baseline_features.shape} (+{baseline_features.shape[1] - df.shape[1]} features)")
    print(f"Advanced features shape: {advanced_features.shape} (+{advanced_features.shape[1] - df.shape[1]} features)")
    print("=" * 50)
    
    # Show some example engineered features
    n_samples = min(5, len(advanced_features))
    print(f"\nSample of engineered features (first {n_samples} rows):")
    
    # Select a subset of interesting features
    interesting_features = [
        col for col in advanced_features.columns 
        if col not in df.columns and not col.endswith('_outlier')
    ][:10]
    
    print(advanced_features[interesting_features].head(n_samples))


if __name__ == "__main__":
    main()