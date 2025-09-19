import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings

class FraudFeatureEngineer:
    """Feature engineering pipeline for fraud detection."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scalers = {}
        self.feature_stats = {}
        self.pca_components = None
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from the Time column."""
        df_copy = df.copy()
        
        # Convert seconds to hours (assuming Time is in seconds)
        df_copy['Hour'] = (df_copy['Time'] / 3600) % 24
        
        # Cyclical encoding of hour
        df_copy['Hour_sin'] = np.sin(2 * np.pi * df_copy['Hour'] / 24)
        df_copy['Hour_cos'] = np.cos(2 * np.pi * df_copy['Hour'] / 24)
        
        # Day of transaction (assuming consecutive days)
        df_copy['Day'] = (df_copy['Time'] / (24 * 3600)).astype(int)
        
        # Time since first transaction
        df_copy['Time_normalized'] = df_copy['Time'] - df_copy['Time'].min()
        
        return df_copy
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
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
        
        return df_copy
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from existing V columns."""
        df_copy = df.copy()
        
        # V columns (PCA features)
        v_columns = [col for col in df.columns if col.startswith('V')]
        
        if v_columns:
            # Statistical aggregations
            df_copy['V_mean'] = df_copy[v_columns].mean(axis=1)
            df_copy['V_std'] = df_copy[v_columns].std(axis=1)
            df_copy['V_min'] = df_copy[v_columns].min(axis=1)
            df_copy['V_max'] = df_copy[v_columns].max(axis=1)
            df_copy['V_range'] = df_copy['V_max'] - df_copy['V_min']
            
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
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between Amount and Time."""
        df_copy = df.copy()
        
        # Amount-Time interactions
        df_copy['Amount_Time_ratio'] = df_copy['Amount'] / (df_copy['Time_normalized'] + 1)
        
        # Amount by hour patterns
        if 'Hour' in df_copy.columns:
            df_copy['Amount_Hour_interaction'] = df_copy['Amount'] * df_copy['Hour']
        
        # High amount at unusual times (night hours)
        if 'Hour' in df_copy.columns:
            night_hours = ((df_copy['Hour'] >= 22) | (df_copy['Hour'] <= 6)).astype(int)
            df_copy['High_amount_night'] = (df_copy['Amount'] > df_copy['Amount'].quantile(0.9)) & night_hours
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [10, 50, 100]) -> pd.DataFrame:
        """Create rolling window features (requires sorted data by time)."""
        df_copy = df.copy().sort_values('Time')
        
        for window in window_sizes:
            # Rolling mean and std of amount
            df_copy[f'Amount_rolling_mean_{window}'] = (
                df_copy['Amount'].rolling(window=window, min_periods=1).mean()
            )
            df_copy[f'Amount_rolling_std_{window}'] = (
                df_copy['Amount'].rolling(window=window, min_periods=1).std()
            )
            
            # Rolling fraud rate (if we have the target variable)
            if 'Class' in df_copy.columns:
                df_copy[f'Fraud_rate_rolling_{window}'] = (
                    df_copy['Class'].rolling(window=window, min_periods=1).mean()
                )
        
        return df_copy
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Add outlier detection features."""
        df_copy = df.copy()
        
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['Class', 'Time']]
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_copy[f'{col}_outlier'] = (
                    (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
                ).astype(int)
        
        # Count total outliers per row
        outlier_cols = [col for col in df_copy.columns if col.endswith('_outlier')]
        df_copy['Total_outliers'] = df_copy[outlier_cols].sum(axis=1)
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, scaler_type: str = 'robust') -> pd.DataFrame:
        """Scale numerical features."""
        df_copy = df.copy()
        
        # Define columns to scale
        scale_columns = [
            'Amount', 'Amount_log', 'Amount_sqrt', 'Time_normalized',
            'V_mean', 'V_std', 'V_range', 'V_entropy'
        ]
        
        # Add V columns if they exist
        v_columns = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        scale_columns.extend(v_columns)
        
        # Filter existing columns
        scale_columns = [col for col in scale_columns if col in df_copy.columns]
        
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        if scale_columns:
            df_copy[scale_columns] = scaler.fit_transform(df_copy[scale_columns])
            self.scalers[scaler_type] = scaler
        
        return df_copy
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        include_time: bool = True,
        include_amount: bool = True,
        include_statistical: bool = True,
        include_interactions: bool = True,
        include_rolling: bool = False,
        include_outliers: bool = True,
        scale: bool = True,
        scaler_type: str = 'robust'
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input dataframe
            include_time: Whether to create time-based features
            include_amount: Whether to create amount-based features
            include_statistical: Whether to create statistical features
            include_interactions: Whether to create interaction features
            include_rolling: Whether to create rolling features (requires sorted data)
            include_outliers: Whether to detect outliers
            scale: Whether to scale features
            scaler_type: Type of scaler ('robust' or 'standard')
        
        Returns:
            Dataframe with engineered features
        """
        df_engineered = df.copy()
        
        if include_time:
            df_engineered = self.create_time_features(df_engineered)
        
        if include_amount:
            df_engineered = self.create_amount_features(df_engineered)
        
        if include_statistical:
            df_engineered = self.create_statistical_features(df_engineered)
        
        if include_interactions:
            df_engineered = self.create_interaction_features(df_engineered)
        
        if include_rolling:
            df_engineered = self.create_rolling_features(df_engineered)
        
        if include_outliers:
            df_engineered = self.detect_outliers(df_engineered)
        
        if scale:
            df_engineered = self.scale_features(df_engineered, scaler_type)
        
        return df_engineered
    
    def get_feature_importance_stats(self, df: pd.DataFrame) -> Dict:
        """Get basic statistics about engineered features."""
        stats = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
        
        return stats


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick baseline feature engineering for rapid prototyping."""
    engineer = FraudFeatureEngineer()
    
    return engineer.engineer_features(
        df,
        include_time=True,
        include_amount=True,
        include_statistical=True,
        include_interactions=False,
        include_rolling=False,
        include_outliers=False,
        scale=True
    )


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering with all features."""
    engineer = FraudFeatureEngineer()
    
    return engineer.engineer_features(
        df,
        include_time=True,
        include_amount=True,
        include_statistical=True,
        include_interactions=True,
        include_rolling=True,
        include_outliers=True,
        scale=True
    )