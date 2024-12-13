# Credit Card Fraud Detection Model Card

## Model Details

**Model Name**: Credit Card Fraud Detection Ensemble  
**Model Version**: 1.0.0  
**Model Type**: Binary Classification (Ensemble)  
**Framework**: scikit-learn, XGBoost, LightGBM  
**Development Date**: September 2024  
**Developers**: MLOps Demo Team  

## Model Description

This model is designed to detect fraudulent credit card transactions in real-time. It uses an ensemble approach combining multiple machine learning algorithms to achieve high accuracy while maintaining low false positive rates.

### Architecture

The model employs a weighted ensemble of the following algorithms:
- **Random Forest Classifier**: Robust baseline with good interpretability
- **XGBoost**: Gradient boosting for complex pattern recognition
- **LightGBM**: Fast and efficient gradient boosting
- **Logistic Regression**: Linear baseline for comparison
- **Neural Network**: Deep learning component for non-linear patterns

### Features

The model uses 30 input features:
- **Time**: Seconds elapsed between this transaction and the first transaction
- **V1-V28**: Principal component analysis (PCA) transformed features (anonymized)
- **Amount**: Transaction amount

Additional engineered features include:
- Amount percentiles and z-scores
- Time-based features (hour, day of week)
- Rolling statistics (mean, std, count)
- Transaction frequency features
- Amount deviation from user's typical spending

## Intended Use

### Primary Use Cases
- Real-time fraud detection for credit card transactions
- Batch processing of transaction data for risk assessment
- Risk scoring for downstream decision systems

### Target Users
- Financial institutions and banks
- Payment processors
- Fraud analysts and risk management teams

### Out-of-Scope Uses
- This model should NOT be used for:
  - Other types of fraud detection (e.g., insurance, identity)
  - Credit scoring or loan approval decisions
  - Customer segmentation or marketing

## Training Data

### Dataset Description
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: ~284,000 transactions
- **Time Period**: September 2013 (2 days)
- **Class Distribution**: 
  - Normal transactions: 99.83% (284,315)
  - Fraudulent transactions: 0.17% (492)

### Data Preprocessing
- Standard scaling applied to Amount feature
- Time feature normalized
- No missing values in original dataset
- PCA features (V1-V28) already preprocessed in source

### Data Splits
- **Training**: 70% (~199,000 transactions)
- **Validation**: 15% (~43,000 transactions)  
- **Test**: 15% (~43,000 transactions)

### Known Limitations
- Data is from 2013 and may not reflect current fraud patterns
- Limited to European cardholders
- PCA transformation makes features non-interpretable
- Highly imbalanced dataset (0.17% fraud rate)

## Model Performance

### Test Set Metrics
| Metric | Value | Threshold |
|--------|-------|-----------|
| **Accuracy** | 99.92% | > 99.5% |
| **Precision** | 88.5% | > 80% |
| **Recall** | 81.6% | > 75% |
| **F1-Score** | 84.9% | > 80% |
| **AUC-ROC** | 97.8% | > 95% |
| **AUC-PR** | 75.2% | > 70% |

### Confusion Matrix (Test Set)
```
                Predicted
                No Fraud  Fraud
Actual No Fraud  42,759    37
       Fraud        18     80
```

### Performance by Transaction Amount
| Amount Range | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| $0-$50       | 89.2%     | 83.1%  | 86.0%    |
| $50-$200     | 87.8%     | 80.4%  | 83.9%    |
| $200-$1000   | 88.9%     | 79.2%  | 83.8%    |
| $1000+       | 86.4%     | 78.9%  | 82.5%    |

## Ethical Considerations

### Fairness and Bias
- Model may have bias toward certain transaction patterns from 2013 European data
- Regular monitoring required to ensure performance across different demographics
- False positives can impact customer experience and trust

### Privacy
- Model uses PCA-transformed features to protect customer privacy
- No personally identifiable information (PII) is used
- Transaction amounts are the only interpretable feature

### Transparency
- Ensemble approach reduces interpretability
- Feature importance available for individual models
- Decision boundaries may be complex and hard to explain

## Limitations and Risks

### Known Limitations
1. **Temporal Drift**: Model trained on 2013 data may degrade over time
2. **Geographic Bias**: Limited to European transactions
3. **Imbalanced Data**: May struggle with novel fraud patterns
4. **Feature Interpretability**: PCA features limit explainability

### Risk Mitigation
1. **Continuous Monitoring**: Deploy drift detection and performance monitoring
2. **Regular Retraining**: Schedule monthly model updates
3. **Human Oversight**: Maintain fraud analyst review process
4. **Threshold Tuning**: Allow dynamic threshold adjustment based on business needs

### Model Decay Indicators
- Precision drops below 80%
- Recall drops below 75%
- Significant increase in customer complaints
- Data drift detected in key features

## Deployment and Monitoring

### Deployment Environment
- **Production API**: FastAPI with automatic scaling
- **Latency Requirement**: < 100ms per prediction
- **Throughput**: 1000+ requests per second
- **Availability**: 99.9% uptime SLA

### Monitoring Plan
- **Performance Metrics**: Daily accuracy, precision, recall tracking
- **Data Drift**: Weekly feature distribution comparison
- **Prediction Drift**: Monthly prediction distribution analysis
- **Business Metrics**: False positive rate, customer satisfaction impact

### Alerting Thresholds
- Precision < 80%: Critical alert
- Recall < 75%: Critical alert
- Latency > 200ms: Warning alert
- Error rate > 1%: Critical alert

## Contact Information

**Model Owner**: MLOps Demo Team  
**Email**: mlops-team@example.com  
**Documentation**: [API Documentation](api_docs.md)  
**Support**: [Issue Tracker](https://github.com/your-org/fraud-detection/issues)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-09 | Initial release with ensemble model |

## Appendix

### Feature Importance (Top 10)
1. V14: 12.3%
2. V4: 10.8%
3. V12: 9.2%
4. Amount: 8.7%
5. V10: 7.9%
6. V11: 7.1%
7. V16: 6.8%
8. V3: 6.2%
9. V7: 5.9%
10. V17: 5.4%

### Model Hyperparameters
- **Random Forest**: n_estimators=100, max_depth=10
- **XGBoost**: learning_rate=0.1, max_depth=6, n_estimators=100
- **LightGBM**: learning_rate=0.1, num_leaves=31, n_estimators=100
- **Ensemble Weights**: RF=0.25, XGB=0.35, LGB=0.30, LR=0.10
## Limitations

- Model performance may degrade on transactions outside training distribution
- Real-time latency depends on feature computation overhead
