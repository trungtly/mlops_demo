#!/usr/bin/env python3
"""
Exploratory Data Analysis script for credit card fraud detection.
Generated output for 2022 data processing.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load the credit card fraud dataset."""
    csv_path = 'data/raw/creditcard.csv'
    print(f"Loading dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB")
    
    return df

def basic_info(df):
    """Display basic information about the dataset."""
    print("\n" + "="*50)
    print("BASIC DATASET INFORMATION")
    print("="*50)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())

def analyze_class_distribution(df):
    """Analyze and visualize class distribution."""
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    class_counts = df['Class'].value_counts()
    print("Class distribution:")
    print(class_counts)
    print(f"Fraud ratio: {class_counts[1] / len(df):.6f} ({class_counts[1]} out of {len(df)})")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.title('Class Distribution (Fraud vs. Normal)', fontsize=14)
    plt.xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/class_distribution_sample.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved to images/class_distribution_sample.png")
    plt.close()

def analyze_amount_distribution(df):
    """Analyze transaction amount distributions."""
    print("\n" + "="*50)
    print("AMOUNT DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Amount statistics by class
    print("Amount statistics by class:")
    print(df.groupby('Class')['Amount'].describe())
    
    # Create amount distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Box plot
    sns.boxplot(x='Class', y='Amount', data=df, ax=axes[0,0])
    axes[0,0].set_title('Amount Distribution by Class (Box Plot)')
    axes[0,0].set_yscale('log')
    
    # Violin plot
    sns.violinplot(x='Class', y='Amount', data=df, ax=axes[0,1])
    axes[0,1].set_title('Amount Distribution by Class (Violin Plot)')
    axes[0,1].set_yscale('log')
    
    # Histogram for normal transactions
    df[df['Class'] == 0]['Amount'].hist(bins=50, ax=axes[1,0], alpha=0.7, color='green')
    axes[1,0].set_title('Amount Distribution - Normal Transactions')
    axes[1,0].set_xlabel('Amount')
    axes[1,0].set_ylabel('Frequency')
    
    # Histogram for fraud transactions
    df[df['Class'] == 1]['Amount'].hist(bins=50, ax=axes[1,1], alpha=0.7, color='red')
    axes[1,1].set_title('Amount Distribution - Fraudulent Transactions')
    axes[1,1].set_xlabel('Amount')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('images/amount_transformations_sample.png', dpi=300, bbox_inches='tight')
    print("Amount distribution plots saved to images/amount_transformations_sample.png")
    plt.close()

def analyze_time_features(df):
    """Analyze time-based features."""
    print("\n" + "="*50)
    print("TIME FEATURE ANALYSIS")
    print("="*50)
    
    # Time statistics
    print("Time statistics:")
    print(df['Time'].describe())
    
    # Create time-based features
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Day'] = (df['Time'] / (3600 * 24)) % 7
    
    # Plot time distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time vs Amount scatter
    fraud_data = df[df['Class'] == 1]
    normal_data = df[df['Class'] == 0].sample(n=min(10000, len(df[df['Class'] == 0])))
    
    axes[0,0].scatter(normal_data['Time'], normal_data['Amount'], alpha=0.5, s=1, c='green', label='Normal')
    axes[0,0].scatter(fraud_data['Time'], fraud_data['Amount'], alpha=0.8, s=2, c='red', label='Fraud')
    axes[0,0].set_title('Time vs Amount Distribution')
    axes[0,0].set_xlabel('Time (seconds)')
    axes[0,0].set_ylabel('Amount')
    axes[0,0].legend()
    
    # Hour distribution
    sns.countplot(x='Hour', hue='Class', data=df.sample(n=min(50000, len(df))), ax=axes[0,1])
    axes[0,1].set_title('Transaction Count by Hour')
    
    # Day distribution
    sns.countplot(x='Day', hue='Class', data=df.sample(n=min(50000, len(df))), ax=axes[1,0])
    axes[1,0].set_title('Transaction Count by Day')
    
    # Time distribution histogram
    df['Time'].hist(bins=50, ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title('Overall Time Distribution')
    axes[1,1].set_xlabel('Time (seconds)')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('images/time_amount_sample.png', dpi=300, bbox_inches='tight')
    print("Time analysis plots saved to images/time_amount_sample.png")
    plt.close()

def analyze_feature_correlations(df):
    """Analyze feature correlations."""
    print("\n" + "="*50)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlations with target
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)
    print("Top 10 features correlated with Class:")
    print(correlations.head(10))
    
    # Create correlation heatmap for top features
    top_features = correlations.head(15).index.tolist()
    corr_matrix = df[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix - Top 15 Features')
    plt.tight_layout()
    plt.savefig('images/features_correlation_sample.png', dpi=300, bbox_inches='tight')
    print("Feature correlation heatmap saved to images/features_correlation_sample.png")
    plt.close()

def analyze_feature_distributions(df):
    """Analyze distributions of V features."""
    print("\n" + "="*50)
    print("V FEATURE DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Select a few V features for detailed analysis
    v_features = ['V1', 'V2', 'V3', 'V4', 'V10', 'V11', 'V12', 'V14', 'V17']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(v_features):
        # Create distribution plots
        normal_data = df[df['Class'] == 0][feature]
        fraud_data = df[df['Class'] == 1][feature]
        
        axes[i].hist(normal_data, bins=50, alpha=0.7, label='Normal', color='green', density=True)
        axes[i].hist(fraud_data, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/v1_v2_distributions.png', dpi=300, bbox_inches='tight')
    print("V feature distributions saved to images/v1_v2_distributions.png")
    plt.close()

def generate_summary_report(df):
    """Generate a summary report."""
    print("\n" + "="*50)
    print("SUMMARY REPORT - EDA FINDINGS")
    print("="*50)
    
    class_counts = df['Class'].value_counts()
    fraud_ratio = class_counts[1] / len(df)
    
    print(f"""
Dataset Overview:
- Total transactions: {len(df):,}
- Total features: {df.shape[1]}
- Fraudulent transactions: {class_counts[1]:,}
- Normal transactions: {class_counts[0]:,}
- Fraud ratio: {fraud_ratio:.4f} ({fraud_ratio*100:.2f}%)

Key Insights:
1. Highly imbalanced dataset - only {fraud_ratio*100:.2f}% fraud cases
2. No missing values detected
3. Amount ranges from ${df['Amount'].min():.2f} to ${df['Amount'].max():,.2f}
4. Time spans {df['Time'].max()/3600:.1f} hours
5. V features are PCA-transformed and anonymized

Recommendations for Model Training:
- Use appropriate sampling techniques (SMOTE, undersampling)
- Consider cost-sensitive learning approaches
- Focus on precision-recall metrics rather than accuracy
- Cross-validation strategy should maintain class balance
    """)

def main():
    """Main execution function."""
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    # Run analysis
    basic_info(df)
    analyze_class_distribution(df)
    analyze_amount_distribution(df)
    analyze_time_features(df)
    analyze_feature_correlations(df)
    analyze_feature_distributions(df)
    generate_summary_report(df)
    
    print("\n" + "="*60)
    print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
# EDA pipeline: generates visualizations and statistical summaries
