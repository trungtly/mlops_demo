"""Data preprocessing module for fraud detection."""

import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample

from ..config import config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing and feature engineering."""
    
    def __init__(self):
        self.target_column = config.TARGET_COLUMN
        self.time_column = config.TIME_COLUMN
        self.amount_column = config.AMOUNT_COLUMN
        self.random_seed = config.RANDOM_SEED
        
        self.scaler = None
        self.feature_columns = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
        
        # Handle missing values (if any)
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(f"Found {missing_before} missing values")
            # For now, drop rows with missing values
            df_clean = df_clean.dropna()
            logger.info(f"Removed rows with missing values. New shape: {df_clean.shape}")
        
        return df_clean
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df_features = df.copy()
        
        # Convert time to hours (assuming time is in seconds)
        df_features['Hour'] = (df_features[self.time_column] / 3600) % 24
        
        # Create time-based categorical features
        df_features['Hour_sin'] = np.sin(2 * np.pi * df_features['Hour'] / 24)
        df_features['Hour_cos'] = np.cos(2 * np.pi * df_features['Hour'] / 24)
        
        # Time periods
        df_features['Is_Night'] = ((df_features['Hour'] >= 23) | (df_features['Hour'] <= 5)).astype(int)
        df_features['Is_Weekend'] = 0  # This would require date information
        
        logger.info("Created time-based features")
        return df_features
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        df_features = df.copy()
        
        # Log transform of amount (add 1 to handle zeros)
        df_features['Amount_log'] = np.log1p(df_features[self.amount_column])
        
        # Amount quantiles
        df_features['Amount_quantile'] = pd.qcut(
            df_features[self.amount_column], 
            q=10, 
            labels=False, 
            duplicates='drop'
        )
        
        # Amount anomaly score (simple z-score)
        df_features['Amount_zscore'] = np.abs(
            (df_features[self.amount_column] - df_features[self.amount_column].mean()) 
            / df_features[self.amount_column].std()
        )
        
        logger.info("Created amount-based features")
        return df_features
    
    def scale_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        # Get feature columns (exclude target and time)
        feature_cols = [col for col in df.columns 
                       if col not in [self.target_column, self.time_column]]
        
        if fit_scaler:
            # Use RobustScaler for better handling of outliers
            self.scaler = RobustScaler()
            df_scaled[feature_cols] = self.scaler.fit_transform(df_scaled[feature_cols])
            self.feature_columns = feature_cols
            logger.info("Fitted and applied scaler to features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            df_scaled[feature_cols] = self.scaler.transform(df_scaled[feature_cols])
            logger.info("Applied existing scaler to features")
        
        return df_scaled
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=config.TEST_SIZE,
            stratify=df[self.target_column],
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),  # Adjust for remaining data
            stratify=train_val[self.target_column],
            random_state=self.random_seed
        )
        
        logger.info(f"Data split - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        logger.info(f"Fraud rates - Train: {train[self.target_column].mean():.4f}, "
                   f"Val: {val[self.target_column].mean():.4f}, "
                   f"Test: {test[self.target_column].mean():.4f}")
        
        return train, val, test
    
    def handle_imbalance(self, 
                        df: pd.DataFrame, 
                        method: str = "none") -> pd.DataFrame:
        """Handle class imbalance."""
        if method == "none":
            return df
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        if method == "undersample":
            # Undersample majority class
            df_majority = df[df[self.target_column] == 0]
            df_minority = df[df[self.target_column] == 1]
            
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=len(df_minority),
                random_state=self.random_seed
            )
            
            df_balanced = pd.concat([df_majority_downsampled, df_minority])
            logger.info(f"Undersampling applied. New shape: {df_balanced.shape}")
            
        elif method == "oversample":
            # Oversample minority class
            df_majority = df[df[self.target_column] == 0]
            df_minority = df[df[self.target_column] == 1]
            
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=self.random_seed
            )
            
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            logger.info(f"Oversampling applied. New shape: {df_balanced.shape}")
        
        else:
            raise ValueError(f"Unknown imbalance method: {method}")
        
        return df_balanced.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
    
    def preprocess(self, 
                  df: pd.DataFrame, 
                  fit_preprocessor: bool = True,
                  imbalance_method: str = "none") -> Dict[str, pd.DataFrame]:
        """Run complete preprocessing pipeline."""
        # Clean data
        df_clean = self.clean_data(df)
        
        # Create features
        df_features = self.create_time_features(df_clean)
        df_features = self.create_amount_features(df_features)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df_features)
        
        # Scale features (fit on training data only)
        train_df = self.scale_features(train_df, fit_scaler=fit_preprocessor)
        val_df = self.scale_features(val_df, fit_scaler=False)
        test_df = self.scale_features(test_df, fit_scaler=False)
        
        # Handle imbalance (only on training data)
        if imbalance_method != "none":
            train_df = self.handle_imbalance(train_df, method=imbalance_method)
        
        result = {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }
        
        logger.info("Preprocessing completed successfully")
        return result
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get data for feature importance analysis."""
        if self.feature_columns is None:
            raise ValueError("Preprocessing must be run first")
        
        return {
            "feature_names": self.feature_columns,
            "correlations": df[self.feature_columns + [self.target_column]].corr()[self.target_column].to_dict()
        }


def main():
    """Main function for data preprocessing."""
    logging.basicConfig(level=logging.INFO)
    
    # This would typically load from the ingestion step
    from .ingestion import DataIngestion
    
    ingestor = DataIngestion()
    df, _ = ingestor.run()
    
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess(df, imbalance_method="none")
    
    print("Preprocessing completed:")
    for split_name, split_df in result.items():
        fraud_rate = split_df[config.TARGET_COLUMN].mean()
        print(f"{split_name}: {split_df.shape}, fraud rate: {fraud_rate:.4f}")


if __name__ == "__main__":
    main()