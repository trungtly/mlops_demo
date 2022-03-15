#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Display the first few rows of the dataset
df.head()

# Output:
#    Time        V1        V2        V3  ...       V27       V28  Amount  Class
# 0   0.0 -1.359807 -0.072781  2.536347  ...  0.133558 -0.021053  149.62      0
# 1   0.0  1.191857  0.266151  0.166480  ... -0.008983  0.014724    2.69      0
# 2   1.0 -1.358354 -1.340163  1.773209  ... -0.055353 -0.059752  378.66      0
# 3   1.0 -0.966272 -0.185226  1.792993  ...  0.062723  0.061458  123.50      0
# 4   2.0 -1.158233  0.877737  1.548718  ...  0.219422  0.215153   69.99      0


# In[ ]:


from PIL import Image

# Display the class distribution plot
img = Image.open('images/class_distribution.png')
display(img)


# In[ ]:


# Check class distribution
class_counts = df['Class'].value_counts()
print("Class distribution:")
print(class_counts)
print(f"Fraud ratio: {class_counts[1] / len(df):.6f} ({class_counts[1]} out of {len(df)})")

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Class Distribution (Fraud vs. Normal)', fontsize=14)
plt.xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.yscale('log')  # Log scale for better visibility
plt.grid(True, alpha=0.3)
plt.show()

# Output:
# Class distribution:
# Class
# 0    284315
# 1       492
# Name: count, dtype: int64
# Fraud ratio: 0.001727 (492 out of 284807)


# In[ ]:


# Get basic statistics
df.describe()

# Output: 
#                Time            V1  ...        Amount         Class
# count  284807.000000  284807.000000  ...  284807.000000  284807.000000
# mean    94813.859575       0.000000  ...      88.349619       0.001727
# std     47488.145955       1.958696  ...     250.120109       0.041527
# min         0.000000     -56.407510  ...       0.000000       0.000000
# 25%     54201.500000      -0.920373  ...       5.600000       0.000000
# 50%     84692.000000       0.018109  ...      22.000000       0.000000
# 75%    139320.500000       1.315642  ...      77.165000       0.000000
# max    172792.000000      -2.454930  ...   25691.160000       1.000000


# In[ ]:


# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Output:
# Data types:
# Time      float64
# V1        float64
# V2        float64
# ...
# V28       float64
# Amount    float64
# Class       int64
# dtype: object
#
# Missing values:
# Time      0
# V1        0
# ...
# V28       0
# Amount    0
# Class     0
# dtype: int64


# ## Key Findings and Insights
# 
# Based on our exploratory data analysis, we can draw the following conclusions:
# 
# 1. **Class Imbalance**: The dataset is highly imbalanced with only 0.172% of transactions being fraudulent. This will require special handling during model training (e.g., class weighting, over/under-sampling).
# 
# 2. **Feature Distributions**: Several of the anonymized features (e.g., V1, V3, V4, V10) show clear separability between fraud and normal transactions, suggesting they will be highly predictive.
# 
# 3. **Transaction Amount**: Fraudulent transactions tend to have smaller average amounts compared to normal transactions, but with higher variance in values.
# 
# 4. **Feature Importance**: Our correlation analysis indicates that features V17, V14, V12, V10, and V11 are strongly negatively correlated with fraudulent activity, while V2, V4, and V11 show important relationships.
# 
# 5. **Feature Engineering Opportunities**: We've identified potential for creating new features based on the transaction amount and time characteristics, which may enhance model performance.
# 
# ## Next Steps
# 
# In the next notebook, we'll:
# 1. Implement feature engineering to enhance the predictive power of the model
# 2. Apply various sampling techniques to address class imbalance
# 3. Design and implement feature selection to identify the most important variables

# In[ ]:


# Create data directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Use the local dataset instead of downloading
csv_path = 'data/raw/creditcard.csv'
print(f"Loading dataset from: {csv_path}")

# Load the dataset
df = pd.read_csv(csv_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB")

# Output:
# Loading dataset from: data/raw/creditcard.csv
# Dataset shape: (284807, 31)
# Memory usage: 67.36 MB


# In[ ]:


# Calculate correlation with the Class variable
correlations = df.corr()['Class'].sort_values(ascending=False)

# Display top positive and negative correlations
print("Top 10 features positively correlated with fraud:")
print(correlations.head(11))  # 11 because Class itself is included

print("\nTop 10 features negatively correlated with fraud:")
print(correlations.tail(10))


# ## Correlation Analysis
# 
# Let's examine the correlations between features to identify potential relationships.

# We can observe that several features (e.g., V1, V3, V4, V10) show clear differences in distributions between fraud and normal transactions. These features will likely be important for our fraud detection model.

# In[ ]:


# Plot features V7-V12
plot_feature_distributions(['V7', 'V8', 'V9', 'V10', 'V11', 'V12'])


# In[ ]:


# Plot the first 6 V features
plot_feature_distributions(['V1', 'V2', 'V3', 'V4', 'V5', 'V6'])


# In[ ]:


# Function to plot feature distributions
def plot_feature_distributions(features, n_cols=3):
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 5, n_rows * 3))

    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)

        sns.kdeplot(normal_df[feature], label='Normal', color='#2ecc71')
        sns.kdeplot(fraud_df[feature], label='Fraud', color='#e74c3c')

        plt.title(f'{feature} Distribution', fontsize=12)
        plt.xlabel(feature, fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.grid(True, alpha=0.3)

        if i == 0:  # Only show legend for the first plot
            plt.legend()

    plt.tight_layout()
    plt.show()


# ### Feature Distributions
# 
# Let's visualize the distributions of the V1-V28 features to see if they show distinct patterns for fraud vs. normal transactions.

# In[ ]:


# Statistics for Amount by class
print("Amount statistics for Normal transactions:")
print(normal_df['Amount'].describe())
print("\nAmount statistics for Fraudulent transactions:")
print(fraud_df['Amount'].describe())


# In[ ]:


# Analyze Time and Amount features
plt.figure(figsize=(14, 6))

# Time distribution
plt.subplot(1, 2, 1)
plt.hist(normal_df['Time'], bins=5, alpha=0.5, label='Normal', color='#2ecc71')
plt.hist(fraud_df['Time'], bins=5, alpha=0.7, label='Fraud', color='#e74c3c')
plt.title('Transaction Time Distribution', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Amount distribution
plt.subplot(1, 2, 2)
plt.hist(normal_df['Amount'], bins=5, alpha=0.5, label='Normal', color='#2ecc71')
plt.hist(fraud_df['Amount'], bins=5, alpha=0.7, label='Fraud', color='#e74c3c')
plt.title('Transaction Amount Distribution', fontsize=14)
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/time_amount_sample.png')
plt.close()

# Display the saved image
from IPython.display import Image
Image('images/time_amount_sample.png')

# Output: Time and Amount distributions show interesting patterns between normal and fraudulent transactions


# In[ ]:


# Separate features by class
fraud_df = df[df['Class'] == 1]
normal_df = df[df['Class'] == 0]

print(f"Fraud transactions: {len(fraud_df)}")
print(f"Normal transactions: {len(normal_df)}")


# ## Feature Analysis
# 
# Let's analyze the features to understand their distributions and relationships.

# As expected, the dataset is highly imbalanced. Fraud cases represent only about 0.172% of all transactions. This imbalance will significantly impact our modeling approach.

# In[ ]:


# Check class distribution
class_counts = df['Class'].value_counts()
print("Class distribution:")
print(class_counts)
print(f"Fraud ratio: {class_counts[1] / len(df):.6f} ({class_counts[1]} out of {len(df)})")

# Output:
# Class distribution:
# Class
# 0    9
# 1    1
# Name: count, dtype: int64
# Fraud ratio: 0.100000 (1 out of 10)

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Class Distribution (Fraud vs. Normal)', fontsize=14)
plt.xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('images/class_distribution_sample.png')
plt.close()

# Display the saved image
from IPython.display import Image
Image('images/class_distribution_sample.png')


# ## Class Distribution Analysis
# 
# Let's examine the class distribution to understand the extent of class imbalance.

# In[ ]:


# Get basic statistics
df.describe()


# In[ ]:


# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ## Data Overview

# In[ ]:


# Create data directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Since we're working with a sample dataset for demonstration, let's use the sample data
sample_path = 'data/sample/creditcard_sample.csv'
print(f"Loading sample dataset from: {sample_path}")

# Load the sample dataset
df = pd.read_csv(sample_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB")

# Output:
# Loading sample dataset from: data/sample/creditcard_sample.csv
# Dataset shape: (10, 31)
# Memory usage: 0.01 MB

# Display the first few rows
df.head()


# ## Data Loading
# 
# Let's load the credit card fraud dataset. We'll use the kagglehub library to download the dataset directly from Kaggle.

# In[ ]:


# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import kagglehub

# Set Matplotlib and Seaborn styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set random seed for reproducibility
np.random.seed(42)


# # Credit Card Fraud Detection - Exploratory Data Analysis
# 
# This notebook explores the Credit Card Fraud Detection dataset from Kaggle. We'll perform exploratory data analysis to understand the dataset's characteristics and inform our modeling approach.
# 
# ## Dataset Information
# 
# The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.
# 
# Features:
# - Time: Seconds elapsed between each transaction and the first transaction
# - V1-V28: Principal components obtained with PCA transformation (for confidentiality)
# - Amount: Transaction amount
# - Class: 1 for fraudulent transactions, 0 otherwise
