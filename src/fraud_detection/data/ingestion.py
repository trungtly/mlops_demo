"""Data ingestion module for credit card fraud detection."""

import logging
from pathlib import Path
from typing import Tuple

import kagglehub
import pandas as pd

from ..config import config

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handle data download and initial loading."""
    
    def __init__(self):
        self.raw_data_dir = config.RAW_DATA_DIR
        self.dataset_name = "mlg-ulb/creditcardfraud"
        
    def download_dataset(self) -> Path:
        """Download dataset from Kaggle using kagglehub."""
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to: {path}")
            return Path(path)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def copy_to_raw_data(self, source_path: Path) -> Path:
        """Copy downloaded data to raw data directory."""
        source_file = source_path / "creditcard.csv"
        target_file = self.raw_data_dir / "creditcard.csv"
        
        if source_file.exists():
            # Copy file to raw data directory
            import shutil
            shutil.copy2(source_file, target_file)
            logger.info(f"Data copied to: {target_file}")
            return target_file
        else:
            raise FileNotFoundError(f"Source file not found: {source_file}")
    
    def load_raw_data(self, file_path: Path = None) -> pd.DataFrame:
        """Load raw data from CSV file."""
        if file_path is None:
            file_path = self.raw_data_dir / "creditcard.csv"
        
        if not file_path.exists():
            logger.warning("Raw data not found. Attempting to download...")
            downloaded_path = self.download_dataset()
            file_path = self.copy_to_raw_data(downloaded_path)
        
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        return df
    
    def get_basic_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset."""
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[config.TARGET_COLUMN].value_counts().to_dict(),
            "fraud_percentage": df[config.TARGET_COLUMN].mean() * 100,
        }
        
        logger.info(f"Dataset info: {info}")
        return info
    
    def run(self) -> Tuple[pd.DataFrame, dict]:
        """Run complete data ingestion pipeline."""
        df = self.load_raw_data()
        info = self.get_basic_info(df)
        return df, info


def main():
    """Main function for data ingestion."""
    logging.basicConfig(level=logging.INFO)
    
    ingestor = DataIngestion()
    df, info = ingestor.run()
    
    print(f"Data ingestion completed. Shape: {df.shape}")
    print(f"Fraud percentage: {info['fraud_percentage']:.3f}%")


if __name__ == "__main__":
    main()