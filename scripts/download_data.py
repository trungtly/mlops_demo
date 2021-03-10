#!/usr/bin/env python3
"""
Data download script for fraud detection dataset.
Usage: python download_data.py --output-dir data/raw/
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from fraud_detection.data.ingestion import FraudDataIngestion
from fraud_detection.config import config


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download fraud detection dataset')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/',
        help='Directory to save downloaded data'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['creditcard', 'synthetic'],
        default='creditcard',
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate downloaded data'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def validate_data(df, dataset_type):
    """Basic validation of downloaded data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating downloaded data")
    
    if dataset_type == 'creditcard':
        # Expected structure for credit card fraud dataset
        expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        
        # Check columns
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        
        # Check basic statistics
        if df.shape[0] < 100000:  # Should have ~280k rows
            logger.warning(f"Dataset seems small: {df.shape[0]} rows")
        
        if not (0 <= df['Class'].mean() <= 0.01):  # Should be ~0.17% fraud
            logger.warning(f"Unexpected fraud rate: {df['Class'].mean():.4f}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Fraud rate: {df['Class'].mean():.4f}")
        logger.info(f"Missing values: {missing_values}")
        
    return True


def main():
    """Main download function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting data download")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data ingestion
        data_ingestion = FraudDataIngestion()
        
        # Run the full ingestion pipeline
        df, info = data_ingestion.run_full_ingestion(
            dataset_type=args.dataset,
            force_download=args.force_download
        )
        
        # If validation is requested
        if args.validate:
            validate_data(df, args.dataset)
        
        logger.info("Data download completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Dataset: {args.dataset}")
        
        raw_file_path = Path(config.RAW_DATA_DIR) / f"{args.dataset}.csv"
        print(f"Raw data file: {raw_file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"File size: {raw_file_path.stat().st_size / (1024*1024):.1f} MB")
        
        if 'Class' in df.columns:
            print(f"Fraud rate: {df['Class'].mean():.4f}")
            print(f"Fraud samples: {int(df['Class'].sum())}")
            print(f"Normal samples: {int((df['Class'] == 0).sum())}")
        
        # Print data split information
        print("\nData splits:")
        for split_name, split_path in info['data_splits'].items():
            path = Path(split_path)
            if path.exists():
                print(f"  {split_name}: {split_path} ({path.stat().st_size / (1024*1024):.1f} MB)")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()