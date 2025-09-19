"""Data ingestion module for credit card fraud detection."""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
import os
import random

import kagglehub
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ..config import config

logger = logging.getLogger(__name__)


class FraudDataIngestion:
    """Handle data download and initial loading."""
    
    def __init__(self):
        """Initialize data ingestion class."""
        self.raw_data_dir = Path(config.RAW_DATA_DIR)
        self.dataset_name = "mlg-ulb/creditcardfraud"
        self.processed_data_dir = Path(config.PROCESSED_DATA_DIR)
        
    def download_creditcard_data(self) -> pd.DataFrame:
        """Download credit card fraud dataset from Kaggle using kagglehub."""
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to: {path}")
            
            # Find the CSV file in the downloaded directory
            csv_path = Path(path) / "creditcard.csv"
            if not csv_path.exists():
                # Look for the CSV file
                for file_path in Path(path).glob("**/*.csv"):
                    csv_path = file_path
                    break
                    
            logger.info(f"Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def generate_synthetic_data(self, n_samples=100000, fraud_ratio=0.002) -> pd.DataFrame:
        """Generate synthetic fraud data for testing purposes."""
        logger.info(f"Generating synthetic data with {n_samples} samples and {fraud_ratio:.4f} fraud ratio")
        
        # Number of fraud and normal samples
        n_frauds = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_frauds
        
        # Generate synthetic data with scikit-learn
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=29,  # 29 features: V1-V28 + Amount
            n_informative=15, 
            n_redundant=5,
            n_classes=2,
            n_clusters_per_class=1,
            weights=[1-fraud_ratio, fraud_ratio],
            random_state=42
        )
        
        # Create DataFrame with feature names similar to the original dataset
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add Time feature - seconds elapsed between transactions
        times = np.sort(np.random.uniform(0, 60*60*24*2, n_samples))  # 2 days of transactions
        df['Time'] = times
        
        # Add Class column (target)
        df['Class'] = y
        
        # Standardize Amount (like in the original dataset)
        mean_amount = 100.0
        std_amount = 150.0
        df['Amount'] = np.abs(df['Amount'] * std_amount + mean_amount)
        
        # Make sure we have exactly the right number of frauds
        current_frauds = df['Class'].sum()
        if current_frauds != n_frauds:
            # Adjust some samples to match desired fraud ratio
            adjustment_needed = n_frauds - current_frauds
            if adjustment_needed > 0:
                # Need more frauds
                normal_indices = df[df['Class'] == 0].index.tolist()
                to_change = random.sample(normal_indices, int(adjustment_needed))
                df.loc[to_change, 'Class'] = 1
            else:
                # Need fewer frauds
                fraud_indices = df[df['Class'] == 1].index.tolist()
                to_change = random.sample(fraud_indices, int(abs(adjustment_needed)))
                df.loc[to_change, 'Class'] = 0
        
        logger.info(f"Generated synthetic data shape: {df.shape}")
        logger.info(f"Fraud samples: {df['Class'].sum()}, Fraud ratio: {df['Class'].mean():.6f}")
        
        return df
    
    def load_local_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from local CSV file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        return df
    
    def split_and_save_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                           val_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Split data into train, validation, and test sets and save to processed directory.
        
        Args:
            df: DataFrame with the data
            test_size: Proportion of data for testing (from overall dataset)
            val_size: Proportion of data for validation (from training set)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with paths to saved data files
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Create processed data directory if it doesn't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # First split: separate test set
        df_train_val, df_test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[config.TARGET_COLUMN]
        )
        
        # Second split: separate validation set from training set
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_size / (1 - test_size),  # Adjust validation size
            random_state=random_state,
            stratify=df_train_val[config.TARGET_COLUMN]
        )
        
        # Save datasets to processed directory
        train_path = self.processed_data_dir / "train.csv"
        val_path = self.processed_data_dir / "validation.csv"
        test_path = self.processed_data_dir / "test.csv"
        
        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        # Log data splits info
        logger.info(f"Train set: {df_train.shape}, fraud rate: {df_train[config.TARGET_COLUMN].mean():.6f}")
        logger.info(f"Validation set: {df_val.shape}, fraud rate: {df_val[config.TARGET_COLUMN].mean():.6f}")
        logger.info(f"Test set: {df_test.shape}, fraud rate: {df_test[config.TARGET_COLUMN].mean():.6f}")
        
        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path)
        }
    
    def get_basic_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset."""
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[config.TARGET_COLUMN].value_counts().to_dict(),
            "fraud_percentage": df[config.TARGET_COLUMN].mean() * 100,
        }
        
        logger.info(f"Dataset info: {info}")
        return info
    
    def run_full_ingestion(self, dataset_type: str = "creditcard", 
                           force_download: bool = False) -> Tuple[pd.DataFrame, dict]:
        """
        Run complete data ingestion pipeline.
        
        Args:
            dataset_type: Type of dataset to use ('creditcard' or 'synthetic')
            force_download: Whether to force download even if local files exist
            
        Returns:
            Tuple of DataFrame and dataset info dictionary
        """
        raw_file_path = self.raw_data_dir / f"{dataset_type}.csv"
        
        # Create raw data directory if it doesn't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        if raw_file_path.exists() and not force_download:
            logger.info(f"Using existing file: {raw_file_path}")
            df = self.load_local_data(raw_file_path)
        else:
            if dataset_type == "creditcard":
                df = self.download_creditcard_data()
            elif dataset_type == "synthetic":
                df = self.generate_synthetic_data()
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
                
            # Save the raw data
            logger.info(f"Saving raw data to: {raw_file_path}")
            df.to_csv(raw_file_path, index=False)
        
        # Get info and split data
        info = self.get_basic_info(df)
        split_paths = self.split_and_save_data(df)
        info["data_splits"] = split_paths
        
        return df, info


def main():
    """Main function for data ingestion."""
    logging.basicConfig(level=logging.INFO)
    
    ingestor = FraudDataIngestion()
    
    # Download and process creditcard dataset
    df, info = ingestor.run_full_ingestion(dataset_type="creditcard")
    
    print("\n" + "="*50)
    print("DATA INGESTION SUMMARY")
    print("="*50)
    print(f"Dataset shape: {info['shape']}")
    print(f"Fraud percentage: {info['fraud_percentage']:.4f}%")
    print(f"Train set: {info['data_splits']['train_path']}")
    print(f"Validation set: {info['data_splits']['val_path']}")
    print(f"Test set: {info['data_splits']['test_path']}")
    print("="*50)


if __name__ == "__main__":
    main()