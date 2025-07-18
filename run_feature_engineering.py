#!/usr/bin/env python3
"""
Feature Engineering script for credit card fraud detection.
Generated output for 2022 data processing.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

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
    
    return df

def create_time_features(df):
    """Create time-based features."""
    print("\n" + "="*50)
    print("CREATING TIME-BASED FEATURES")
    print("="*50)
    
    # Convert time to hours and create cyclical features  
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Day'] = (df['Time'] / (3600 * 24)) % 7
    
    # Create cyclical features for hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Create cyclical features for day
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
    
    # Time-based patterns
    df['Is_Weekend'] = (df['Day'] >= 5).astype(int)
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    print(f"Created {6} new time-based features")
    
    return df

def create_amount_features(df):
    """Create amount-based features."""
    print("\n" + "="*50)
    print("CREATING AMOUNT-BASED FEATURES")
    print("="*50)
    
    # Log transformation for amount (adding 1 to handle 0 values)
    df['Amount_log'] = np.log1p(df['Amount'])
    
    # Amount quantiles
    df['Amount_percentile'] = df['Amount'].rank(pct=True)
    
    # High/Low amount flags
    amount_25 = df['Amount'].quantile(0.25)
    amount_75 = df['Amount'].quantile(0.75)
    amount_95 = df['Amount'].quantile(0.95)
    
    df['Is_Low_Amount'] = (df['Amount'] <= amount_25).astype(int)
    df['Is_High_Amount'] = (df['Amount'] >= amount_75).astype(int)
    df['Is_Very_High_Amount'] = (df['Amount'] >= amount_95).astype(int)
    
    # Amount category
    df['Amount_Category'] = pd.cut(df['Amount'], 
                                 bins=[0, 10, 100, 1000, float('inf')],
                                 labels=['Very_Low', 'Low', 'Medium', 'High'])
    
    # One-hot encode amount category
    amount_dummies = pd.get_dummies(df['Amount_Category'], prefix='Amount')
    df = pd.concat([df, amount_dummies], axis=1)
    
    print(f"Created {9} new amount-based features")
    
    return df

def create_interaction_features(df):
    """Create interaction features between V features."""
    print("\n" + "="*50)
    print("CREATING INTERACTION FEATURES")
    print("="*50)
    
    # Top correlated V features with Class
    top_v_features = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 'V11', 'V4']
    
    # Create interaction features between top features
    interactions = []
    for i, feat1 in enumerate(top_v_features[:5]):
        for feat2 in top_v_features[i+1:6]:
            interaction_name = f'{feat1}_{feat2}_interaction'
            df[interaction_name] = df[feat1] * df[feat2]
            interactions.append(interaction_name)
    
    print(f"Created {len(interactions)} interaction features")
    
    return df

def create_aggregate_features(df):
    """Create aggregate features from V features."""
    print("\n" + "="*50)
    print("CREATING AGGREGATE FEATURES")
    print("="*50)
    
    # V feature columns
    v_cols = [col for col in df.columns if col.startswith('V')]
    
    # Aggregate statistics
    df['V_sum'] = df[v_cols].sum(axis=1)
    df['V_mean'] = df[v_cols].mean(axis=1)
    df['V_std'] = df[v_cols].std(axis=1)
    df['V_min'] = df[v_cols].min(axis=1)
    df['V_max'] = df[v_cols].max(axis=1)
    df['V_range'] = df['V_max'] - df['V_min']
    
    # Count of negative and positive V features
    df['V_negative_count'] = (df[v_cols] < 0).sum(axis=1)
    df['V_positive_count'] = (df[v_cols] > 0).sum(axis=1)
    df['V_zero_count'] = (df[v_cols] == 0).sum(axis=1)
    
    print(f"Created 9 aggregate features from V features")
    
    return df

def apply_scaling(df, target_col='Class'):
    """Apply scaling to features."""
    print("\n" + "="*50)
    print("APPLYING FEATURE SCALING")
    print("="*50)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numerical columns (excluding categorical dummy variables)
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove categorical dummy columns from scaling
    categorical_dummies = [col for col in numerical_cols if col.startswith('Amount_')]
    numerical_cols = [col for col in numerical_cols if col not in categorical_dummies]
    
    print(f"Scaling {len(numerical_cols)} numerical features")
    
    # Apply robust scaling (less sensitive to outliers)
    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    print("Feature scaler saved to models/feature_scaler.pkl")
    
    return X_scaled, y, scaler

def select_features(X, y, k=50):
    """Select top k features using statistical tests."""
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)
    
    # Use f_classif for feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'score': feature_scores
    }).sort_values('score', ascending=False)
    
    print(f"Selected top {k} features out of {X.shape[1]} total features")
    print("\\nTop 10 selected features:")
    print(feature_importance.head(10))
    
    # Save feature selector
    joblib.dump(selector, 'models/feature_selector.pkl')
    print("Feature selector saved to models/feature_selector.pkl")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_20_features = feature_importance.head(20)
    sns.barplot(data=top_20_features, x='score', y='feature')
    plt.title('Top 20 Feature Importance Scores')
    plt.xlabel('F-Score')
    plt.tight_layout()
    plt.savefig('images/feature_importance_sample.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved to images/feature_importance_sample.png")
    plt.close()
    
    return pd.DataFrame(X_selected, columns=selected_features), selector

def create_balanced_datasets(X, y):
    """Create balanced datasets using different sampling techniques."""
    print("\n" + "="*50)
    print("CREATING BALANCED DATASETS")
    print("="*50)
    
    balanced_datasets = {}
    
    # Original dataset
    balanced_datasets['original'] = (X, y)
    print(f"Original dataset: {y.value_counts().to_dict()}")
    
    # SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        balanced_datasets['smote'] = (X_smote, y_smote)
        print(f"SMOTE dataset: {y_smote.value_counts().to_dict()}")
    except Exception as e:
        print(f"SMOTE failed: {e}")
    
    # ADASYN
    try:
        adasyn = ADASYN(random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
        balanced_datasets['adasyn'] = (X_adasyn, y_adasyn)
        print(f"ADASYN dataset: {y_adasyn.value_counts().to_dict()}")
    except Exception as e:
        print(f"ADASYN failed: {e}")
    
    # Random Under Sampling
    try:
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X, y)
        balanced_datasets['undersampling'] = (X_rus, y_rus)
        print(f"Under-sampling dataset: {y_rus.value_counts().to_dict()}")
    except Exception as e:
        print(f"Under-sampling failed: {e}")
    
    # SMOTE + Tomek
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X, y)
        balanced_datasets['smote_tomek'] = (X_smote_tomek, y_smote_tomek)
        print(f"SMOTE-Tomek dataset: {y_smote_tomek.value_counts().to_dict()}")
    except Exception as e:
        print(f"SMOTE-Tomek failed: {e}")
    
    return balanced_datasets

def save_processed_data(balanced_datasets):
    """Save all processed datasets."""
    print("\n" + "="*50)
    print("SAVING PROCESSED DATASETS")
    print("="*50)
    
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)
    
    for name, (X, y) in balanced_datasets.items():
        # Combine features and target
        df_processed = X.copy()
        df_processed['Class'] = y
        
        # Save to CSV
        output_path = f'data/processed/creditcard_{name}.csv'
        df_processed.to_csv(output_path, index=False)
        print(f"Saved {name} dataset: {df_processed.shape} -> {output_path}")

def create_train_test_splits(balanced_datasets):
    """Create train/test splits for each balanced dataset."""
    print("\n" + "="*50)
    print("CREATING TRAIN/TEST SPLITS")
    print("="*50)
    
    splits = {}
    
    for name, (X, y) in balanced_datasets.items():
        # Create stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        splits[name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"{name} split - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Train class dist: {y_train.value_counts().to_dict()}")
        print(f"  Test class dist: {y_test.value_counts().to_dict()}")
    
    # Save splits
    os.makedirs('data/splits', exist_ok=True)
    for name, split_data in splits.items():
        for split_name, data in split_data.items():
            output_path = f'data/splits/{name}_{split_name}.pkl'
            joblib.dump(data, output_path)
    
    print("All train/test splits saved to data/splits/")
    
    return splits

def generate_feature_engineering_report(df_original, df_engineered, selected_features):
    """Generate a comprehensive feature engineering report."""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING SUMMARY REPORT")
    print("="*50)
    
    print(f"""
Original Dataset:
- Shape: {df_original.shape}
- Features: {df_original.shape[1] - 1} (excluding target)

After Feature Engineering:
- Shape: {df_engineered.shape}
- Features: {df_engineered.shape[1] - 1} (excluding target)
- New features created: {df_engineered.shape[1] - df_original.shape[1]}

Feature Categories Created:
1. Time-based features: 6 features
   - Hour, Day (cyclical encoding)
   - Weekend and night indicators
   
2. Amount-based features: 9 features
   - Log transformation, percentiles
   - Amount categories and flags
   
3. Interaction features: 10 features
   - Between top V features
   
4. Aggregate features: 9 features
   - Statistics from V features
   
5. Selected features: {len(selected_features)} features
   - Using statistical feature selection
   
Data Balancing Techniques Applied:
- SMOTE (Synthetic Minority Oversampling)
- ADASYN (Adaptive Synthetic Sampling)
- Random Under-sampling
- SMOTE + Tomek Links

Preprocessing Steps:
✓ Feature scaling using RobustScaler
✓ Feature selection using SelectKBest
✓ Class balancing using multiple techniques
✓ Train/test splits with stratification
    """)

def main():
    """Main execution function."""
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - FEATURE ENGINEERING")
    print("="*60)
    print(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    df_original = df.copy()
    
    # Feature engineering steps
    df = create_time_features(df)
    df = create_amount_features(df)
    df = create_interaction_features(df)
    df = create_aggregate_features(df)
    
    # Drop categorical column used for dummy creation
    if 'Amount_Category' in df.columns:
        df = df.drop('Amount_Category', axis=1)
    
    # Apply scaling
    X_scaled, y, scaler = apply_scaling(df)
    
    # Feature selection
    X_selected, selector = select_features(X_scaled, y, k=50)
    
    # Create balanced datasets
    balanced_datasets = create_balanced_datasets(X_selected, y)
    
    # Save processed data
    save_processed_data(balanced_datasets)
    
    # Create train/test splits
    splits = create_train_test_splits(balanced_datasets)
    
    # Generate report
    generate_feature_engineering_report(df_original, df, X_selected.columns.tolist())
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
# Feature engineering: transforms raw features into model-ready format
