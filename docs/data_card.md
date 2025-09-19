# Credit Card Fraud Detection Dataset Card

## Dataset Summary

**Dataset Name**: Credit Card Fraud Detection Dataset  
**Source**: Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Original Provider**: Machine Learning Group - ULB (Université Libre de Bruxelles)  
**License**: Open Data Commons Open Database License (ODbL) v1.0  
**Last Updated**: September 2013  

## Dataset Description

This dataset contains credit card transactions made by European cardholders over a two-day period in September 2013. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

## Data Composition

### Basic Statistics
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Normal Transactions**: 284,315 (99.828%)
- **Features**: 31 (30 input features + 1 target)
- **Time Period**: 48 hours (September 2013)
- **Missing Values**: None

### Feature Description

| Feature | Type | Description | Range/Distribution |
|---------|------|-------------|-------------------|
| **Time** | Numerical | Seconds elapsed between transaction and first transaction | 0 - 172,792 seconds |
| **V1-V28** | Numerical | PCA-transformed features (anonymized) | Normalized, mean ≈ 0 |
| **Amount** | Numerical | Transaction amount | €0.00 - €25,691.16 |
| **Class** | Binary | Target variable (0=Normal, 1=Fraud) | 0 or 1 |

### PCA Features (V1-V28)
- Pre-processed using Principal Component Analysis (PCA)
- Anonymized to protect customer privacy
- Already standardized (mean ≈ 0, std ≈ 1)
- Capture 95%+ of variance in original features
- Not interpretable due to transformation

### Amount Distribution
```
Normal Transactions:
- Mean: €88.35
- Median: €22.00
- Std: €250.12
- 95th percentile: €384.52

Fraudulent Transactions:
- Mean: €122.21
- Median: €9.25
- Std: €256.68
- 95th percentile: €675.37
```

### Time Distribution
- Transactions span exactly 2 days (172,792 seconds)
- Peak hours: 10:00-16:00 and 19:00-23:00 CET
- Fraud distribution varies by time of day
- Higher fraud rates during night hours (00:00-06:00)

## Data Collection

### Collection Process
- **Source**: Real anonymized credit card transactions
- **Collection Period**: September 2013 (48 hours)
- **Geographic Scope**: European cardholders only
- **Anonymization**: PCA transformation applied to sensitive features

### Preprocessing Applied
1. **PCA Transformation**: Original features transformed to V1-V28
2. **Normalization**: PCA features standardized
3. **Time Encoding**: Time as seconds from start
4. **Amount Preservation**: Only Amount and Time kept interpretable

### Data Quality
- ✅ No missing values
- ✅ No duplicate transactions
- ✅ Consistent data types
- ✅ Valid ranges for all features
- ⚠️ Highly imbalanced classes (0.172% fraud)
- ⚠️ Limited time period (2 days only)

## Known Limitations and Biases

### Temporal Limitations
- **Limited Time Span**: Only 2 days of data
- **Seasonal Bias**: September 2013 only (no seasonal variations)
- **Era Bias**: 2013 fraud patterns may differ from current patterns
- **Weekend/Weekday**: May not represent full week patterns

### Geographic and Demographic Limitations
- **Geographic Scope**: European cardholders only
- **Currency**: Euros only
- **Banking Systems**: European banking infrastructure
- **Cultural Patterns**: European spending behaviors

### Technical Limitations
- **Feature Interpretability**: PCA transformation prevents feature interpretation
- **Class Imbalance**: Severe imbalance (492:284,315 ratio)
- **Anonymization**: Limits domain knowledge application
- **Static Nature**: No ability to track customer behavior over time

### Potential Biases
1. **Socioeconomic Bias**: European cardholder demographics
2. **Merchant Bias**: Limited to specific merchant types/regions
3. **Detection Bias**: Only known/detected frauds included
4. **Reporting Bias**: May reflect 2013 fraud detection capabilities

## Use Cases and Applications

### Recommended Use Cases
- ✅ Fraud detection algorithm development
- ✅ Machine learning model benchmarking
- ✅ Imbalanced dataset technique evaluation
- ✅ Academic research on fraud detection
- ✅ Proof-of-concept development

### Not Recommended For
- ❌ Production fraud detection without additional data
- ❌ Non-European market deployment
- ❌ Current fraud pattern analysis
- ❌ Customer behavior analysis
- ❌ Credit scoring or risk assessment

## Ethical Considerations

### Privacy Protection
- **Anonymization**: PCA transformation protects customer identity
- **No PII**: No personally identifiable information included
- **Aggregated Data**: Individual transactions cannot be traced back
- **Compliance**: Adheres to European data protection standards

### Fairness Concerns
- **Geographic Bias**: May not perform well for non-European populations
- **Temporal Bias**: 2013 patterns may disadvantage current fraud types
- **Socioeconomic Bias**: European banking patterns may not generalize

### Responsible Use Guidelines
1. **Validation**: Always validate on current, relevant data before deployment
2. **Monitoring**: Implement continuous monitoring for bias and drift
3. **Transparency**: Disclose dataset limitations to stakeholders
4. **Regular Updates**: Retrain models with more recent data when available

## Dataset Splits

### Standard Splits (Recommended)
```python
# Chronological split (recommended for time series)
- Training: First 60% of timeline (~170K transactions)
- Validation: Next 20% of timeline (~57K transactions)
- Test: Last 20% of timeline (~57K transactions)

# Random stratified split (for model comparison)
- Training: 70% (~200K transactions, ~344 frauds)
- Validation: 15% (~43K transactions, ~74 frauds)
- Test: 15% (~43K transactions, ~74 frauds)
```

### Class Distribution per Split
| Split | Normal | Fraud | Fraud % |
|-------|--------|-------|---------|
| Train | 199,021 | 344 | 0.173% |
| Valid | 42,662 | 74 | 0.173% |
| Test | 42,632 | 74 | 0.173% |

## Data Schema

### File Format
- **Format**: CSV (Comma-separated values)
- **Encoding**: UTF-8
- **Size**: ~150 MB uncompressed
- **Delimiter**: Comma (,)
- **Header**: Yes (first row)

### Schema Definition
```sql
CREATE TABLE transactions (
    Time DECIMAL(10,2),           -- Seconds from start
    V1 DECIMAL(12,8),            -- PCA feature 1
    V2 DECIMAL(12,8),            -- PCA feature 2
    ...
    V28 DECIMAL(12,8),           -- PCA feature 28
    Amount DECIMAL(10,2),        -- Transaction amount
    Class INTEGER                -- 0=Normal, 1=Fraud
);
```

### Data Validation Rules
```python
# Validation constraints
Time: >= 0 and <= 172792
V1-V28: Any real number (typically -5 to +5)
Amount: >= 0 and <= 30000  # Reasonable upper bound
Class: 0 or 1
```

## Version History

| Version | Date | Changes | Size |
|---------|------|---------|------|
| 1.0 | 2016-03 | Initial Kaggle release | 150 MB |
| Current | 2024-09 | Used in this project | 150 MB |

## Access and Usage

### Download Instructions
```bash
# Using Kaggle API
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud

# Using project script
python scripts/download_data.py
```

### Citation
```
@misc{creditcard2013,
  title={Credit Card Fraud Detection Dataset},
  author={Machine Learning Group - ULB},
  year={2013},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}
}
```

## Contact and Support

**Original Data Provider**: Machine Learning Group - ULB  
**Dataset Maintainer**: Kaggle Community  
**Project Contact**: mlops-team@example.com  
**Issues**: [GitHub Issues](https://github.com/your-org/fraud-detection/issues)

## Related Resources

- [Model Card](model_card.md) - Information about models trained on this data
- [API Documentation](api_docs.md) - How to use the fraud detection API
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - Original source
- [Research Paper](https://www.researchgate.net/publication/319867396) - Academic background