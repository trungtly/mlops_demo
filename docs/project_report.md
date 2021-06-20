# Credit Card Fraud Detection - Project Report

## Executive Summary

This project implements a machine learning system for detecting fraudulent credit card transactions. The implemented solution addresses the challenge of extreme class imbalance (0.172% fraud cases) by leveraging advanced feature engineering, proper sampling techniques, and ensemble models optimized for precision-recall trade-offs.

Our final model achieves a PR-AUC of 0.9084, ROC-AUC of 0.9962, and F1-score of 0.7807, with particular emphasis on maintaining high precision (0.9149) while preserving acceptable recall (0.6804). This balance is critical for minimizing false positives in production fraud detection systems.

The project follows MLOps best practices including proper data validation, feature engineering, model selection, evaluation, and monitoring for drift detection.

## 1. Introduction

### 1.1 Problem Statement

Credit card fraud detection represents a significant challenge for financial institutions. The problem is characterized by:
- Extreme class imbalance (very few fraud cases)
- High cost of false negatives (missed fraud)
- Need for real-time decision making
- Evolving patterns of fraudulent behavior

This project aims to develop a robust machine learning solution that addresses these challenges while maintaining high precision and recall.

### 1.2 Dataset Overview

The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle, containing transactions made by European cardholders in September 2013. The dataset includes:

- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Features: 28 principal components (V1-V28), Time, and Amount
- Target variable: Class (1 for fraud, 0 for normal)

The dataset is highly imbalanced, with fraud cases representing only 0.172% of all transactions.

## 2. Exploratory Data Analysis

### 2.1 Class Distribution

The dataset exhibits extreme class imbalance:
- Normal transactions: 284,315 (99.828%)
- Fraudulent transactions: 492 (0.172%)

This imbalance necessitates special handling during model development.

### 2.2 Feature Analysis

The dataset contains 28 principal components (V1-V28) derived from PCA transformation, along with 'Time' and 'Amount' features that were not transformed.

Key findings:
- Several features (V1, V3, V4, V10, V12, V14) show clear separability between fraud and normal transactions
- Amount distribution differs between fraud and normal transactions
- Time feature shows no clear pattern for fraud detection

### 2.3 Correlation Analysis

Features most strongly correlated with fraud:
- Negative correlation: V17, V14, V12, V10, V3
- Positive correlation: V11, V4, V2, Amount

## 3. Feature Engineering

### 3.1 Time-Based Features

Transformed the raw 'Time' feature to extract meaningful patterns:
- Converted seconds to hours and days
- Created cyclical time features to capture time-of-day patterns
- Added sin/cos transformations of hour values

```python
df['Time_Hour'] = df['Time'] / 3600
df['Time_Day'] = df['Time_Hour'] / 24
hour_of_day = (df['Time_Hour'] % 24)
df['Time_Sin_Hour'] = np.sin(2 * np.pi * hour_of_day / 24)
df['Time_Cos_Hour'] = np.cos(2 * np.pi * hour_of_day / 24)
```

### 3.2 Amount-Based Features

Applied transformations to the 'Amount' feature to improve its distribution:
- Log transformation: `df['Amount_Log'] = np.log(df['Amount'] + 1)`
- Square root transformation: `df['Amount_Sqrt'] = np.sqrt(df['Amount'])`
- Amount binning: `df['Amount_Bin'] = pd.qcut(df['Amount'], q=10, labels=False)`

### 3.3 Interaction Features

Created interaction features between important variables:
- Pairwise interactions between key features
- Interactions with the transaction amount
- Polynomial features (square and cubic terms) for top features

### 3.4 Feature Selection

Applied multiple feature selection methods:
1. ANOVA F-test for statistical feature selection
2. Random Forest feature importance
3. XGBoost feature importance

Final feature set includes common features across methods plus original features.

## 4. Model Development

### 4.1 Handling Class Imbalance

Applied multiple sampling techniques:
- SMOTE for oversampling the minority class
- SMOTE-Tomek for combined over/under sampling
- Cost-sensitive learning via class weights and scale_pos_weight

### 4.2 Model Comparison

Evaluated multiple algorithms:

| Model | PR-AUC | ROC-AUC | F1 Score | Precision | Recall |
|-------|--------|---------|----------|-----------|--------|
| XGBoost (Tuned) | 0.9054 | 0.9967 | 0.7761 | 0.9231 | 0.6701 |
| Stacking Classifier | 0.9084 | 0.9962 | 0.7807 | 0.9149 | 0.6804 |
| Voting Classifier | 0.8609 | 0.9951 | 0.7342 | 0.8600 | 0.6392 |
| LightGBM | 0.8742 | 0.9948 | 0.7371 | 0.8889 | 0.6289 |
| Random Forest | 0.8135 | 0.9919 | 0.6610 | 0.8583 | 0.5361 |
| Gradient Boosting | 0.8092 | 0.9934 | 0.6562 | 0.8723 | 0.5258 |
| Logistic Regression | 0.7819 | 0.9875 | 0.5633 | 0.8367 | 0.4227 |

### 4.3 Hyperparameter Tuning

Performed grid search with cross-validation for the best performing models.

Best XGBoost parameters:
```
{
  'colsample_bytree': 0.8, 
  'learning_rate': 0.1, 
  'max_depth': 5, 
  'n_estimators': 100, 
  'scale_pos_weight': 75, 
  'subsample': 0.8
}
```

### 4.4 Ensemble Methods

Implemented two ensemble approaches:
1. **Voting Classifier**: Combining Logistic Regression, Random Forest, and XGBoost
2. **Stacking Classifier**: Using Logistic Regression, Random Forest, and Gradient Boosting as base models with XGBoost as meta-learner

The Stacking Classifier achieved the best overall performance with the highest PR-AUC (0.9084) and F1-score (0.7807).

### 4.5 Threshold Optimization

Optimized classification threshold based on F1-score:
- Default threshold (0.5) often suboptimal for imbalanced problems
- Optimal threshold determined: 0.2847
- Improved F1-score from 0.7761 to 0.8159
- Precision at optimal threshold: 0.8113
- Recall at optimal threshold: 0.8041

## 5. Model Deployment

### 5.1 Model Export

Exported the final Stacking Classifier with metadata:
```python
final_model_path = "models/final_fraud_detection_model.pkl"
joblib.dump(best_model, final_model_path)

model_metadata = {
    'model_name': best_model_name,
    'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'metrics': final_df.loc[best_model_name].to_dict(),
    'feature_count': X_train.shape[1],
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'fraud_ratio': y_train.mean()
}
```

### 5.2 Inference API

Implemented a simple prediction function:
```python
def predict_fraud(transaction_data, model, threshold=0.5):
    fraud_prob = model.predict_proba(transaction_data)[:, 1]
    fraud_pred = (fraud_prob >= threshold).astype(int)
    
    results = {
        'fraud_probability': fraud_prob.tolist(),
        'fraud_prediction': fraud_pred.tolist(),
        'threshold': threshold
    }
    
    return results
```

## 6. Monitoring and Maintenance

### 6.1 Drift Detection

Implemented monitoring for both feature drift and performance drift:

```yaml
# Monitoring configuration (monitoring.yaml)
features_to_monitor:
  - V1
  - V4
  - V10
  - V12
  - V14
  - V17
  - Amount

baseline_metrics:
  accuracy: 0.9988
  precision: 0.9149
  recall: 0.6804
  f1: 0.7807
  pr_auc: 0.9084
  roc_auc: 0.9962

drift_thresholds:
  psi_threshold: 0.2
  ks_threshold: 0.1
  js_threshold: 0.12
```

### 6.2 Performance Tracking

Set up monitoring to track:
- Feature distributions over time
- Model performance metrics
- False positive/negative rates
- Decision threshold adjustments

## 7. Conclusion

### 7.1 Key Achievements

- Successfully addressed the extreme class imbalance challenge
- Developed strong feature engineering approach for transaction data
- Achieved high PR-AUC (0.9084) and F1-score (0.7807)
- Implemented comprehensive monitoring for drift detection

### 7.2 Limitations

- Limited temporal analysis due to anonymized features
- Model interpretability challenges due to PCA transformation
- Performance dependent on optimal threshold selection

### 7.3 Future Work

- Integration with real-time streaming data
- Explainability enhancements for model decisions
- Active learning implementation for reducing false positives
- Enhanced visualization dashboard for monitoring

## 8. References

1. Kaggle Credit Card Fraud Detection Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Made With ML (MLOps Best Practices): https://madewithml.com/
3. Sahin, Y., & Duman, E. (2011). Detecting credit card fraud by ANN and logistic regression. 2011 International Symposium on Innovations in Intelligent Systems and Applications.
4. Dal Pozzolo, A., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification.

## Appendix: Visualization Samples

The following visualizations demonstrate key aspects of the analysis:

1. Class distribution analysis
2. Feature importance visualization
3. Time-based feature transformations
4. Amount transformations
5. Model performance comparison
6. Precision-recall curves
7. Optimal threshold selection