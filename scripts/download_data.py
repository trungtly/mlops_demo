#!/usr/bin/env python3
"""
Data download script for fraud detection dataset.
Usage: python download_data.py --output-dir data/raw/
"""

import argparse
import sys
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from fraud_detection.data.ingestion import FraudDataIngestion
    from fraud_detection.config import config
    FRAUD_MODULE_AVAILABLE = True
except ImportError:
    FRAUD_MODULE_AVAILABLE = False


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


def create_demo_dataset(output_dir):
    """Create a small example dataset for demonstration purposes."""
    # Create the directory structure
    sample_dir = os.path.join(os.path.dirname(output_dir), "sample")
    os.makedirs(sample_dir, exist_ok=True)
    
    print("Creating demo dataset for testing...")
    
    # Create a sample dataset with 10 records
    sample_data = pd.DataFrame({
        'Time': [0, 0, 1, 1, 2, 2, 4, 7, 7, 9],
        'V1': [-1.359807, 1.191857, -1.358354, -0.966272, -1.158233, -0.425966, 1.229658, -0.644269, -0.894286, -0.338262],
        'V2': [-0.072781, 0.266151, -1.340163, -0.185226, 0.877737, 0.960523, 0.141004, 1.417964, 0.286157, 1.119593],
        'V3': [2.536347, 0.166480, 1.773209, 1.792993, 1.548718, 1.141109, 0.045371, 1.074380, -0.113192, 1.044367],
        'V4': [1.378155, 0.448154, 0.379780, -0.863291, 0.403034, -0.168252, 1.202613, -0.492199, -0.271526, -0.222187],
        'V5': [-0.338321, 0.060018, -0.503198, -0.010309, -0.407193, 0.420987, 0.191881, 0.948934, 2.669599, 0.499361],
        'Amount': [149.62, 2.69, 378.66, 123.50, 69.99, 3.67, 52.21, 0.00, 70.73, 59.86],
        'Class': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    })
    
    # Add V6-V28 with random values
    for i in range(6, 29):
        sample_data[f'V{i}'] = np.random.randn(len(sample_data))
    
    # Reorder columns to match the original dataset
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    sample_data = sample_data[cols]
    
    # Save the sample dataset
    sample_output_file = os.path.join(sample_dir, "creditcard_sample.csv")
    sample_data.to_csv(sample_output_file, index=False)
    print(f"Demo dataset saved to {sample_output_file}")
    
    return sample_output_file


def download_with_kagglehub(output_dir, sample=False):
    """Download dataset using kagglehub."""
    if not KAGGLEHUB_AVAILABLE:
        print("kagglehub not installed. Please install it with: pip install kagglehub")
        return None
    
    try:
        print("Downloading Credit Card Fraud Detection dataset...")
        # Download the dataset using kagglehub
        path = kagglehub.model_download("mlg-ulb/creditcardfraud")
        
        # Path to the CSV file in the downloaded dataset
        csv_file = os.path.join(path, "creditcard.csv")
        
        # If the path exists and is a file
        if os.path.exists(csv_file):
            # Copy to the output directory
            output_file = os.path.join(output_dir, "creditcard.csv")
            df = pd.read_csv(csv_file)
            df.to_csv(output_file, index=False)
            print(f"Dataset saved to {output_file}")
            
            # Create a sample dataset if requested
            if sample:
                # Ensure we have enough fraud cases in the sample
                fraud_df = df[df['Class'] == 1].sample(min(10, len(df[df['Class'] == 1])))
                normal_df = df[df['Class'] == 0].sample(min(990, len(df[df['Class'] == 0])))
                sample_df = pd.concat([normal_df, fraud_df])
                sample_df = sample_df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
                
                # Save the sample dataset
                sample_output_dir = os.path.join(os.path.dirname(output_dir), "sample")
                os.makedirs(sample_output_dir, exist_ok=True)
                sample_output_file = os.path.join(sample_output_dir, "creditcard_sample.csv")
                sample_df.to_csv(sample_output_file, index=False)
                print(f"Sample dataset saved to {sample_output_file}")
            
            return output_file
    except Exception as e:
        print(f"Error downloading dataset with kagglehub: {e}")
    
    return None


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
        
        success = False
        
        # Try using the full fraud_detection module if available
        if FRAUD_MODULE_AVAILABLE:
            try:
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
                
                success = True
            except Exception as e:
                logger.error(f"Failed to use FraudDataIngestion: {str(e)}")
                logger.info("Trying fallback methods...")
        
        # If the first method failed, try using kagglehub directly
        if not success:
            output_file = download_with_kagglehub(args.output_dir, sample=True)
            if output_file:
                print(f"Successfully downloaded dataset to {output_file}")
                success = True
        
        # If all else fails, create a demo dataset
        if not success:
            print("Falling back to creating a demo dataset...")
            sample_file = create_demo_dataset(args.output_dir)
            print(f"Created demo dataset at {sample_file}")
            print("For the actual dataset, please download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()# Fix issue with data preprocessing scaling

# Supports kaggle and local file sources
