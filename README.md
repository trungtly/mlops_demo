# Credit Card Fraud Detection - MLOps Demo

A comprehensive MLOps demonstration project for credit card fraud detection, showcasing production-ready machine learning practices including data validation, experiment tracking, model serving, monitoring, and CI/CD automation.

## Project Overview

This project demonstrates end-to-end MLOps practices for building a production-grade fraud detection system:

- **Dataset**: Credit Card Fraud Detection from Kaggle (284,807 transactions, 0.172% fraud rate)
- **Challenge**: Highly imbalanced classification problem
- **Approach**: Cost-sensitive learning with proper evaluation metrics
- **Architecture**: Modular, testable, and scalable design

## Repository Structure

```
├── src/fraud_detection/        # Main source code
│   ├── data/                  # Data ingestion & preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Model definitions
│   ├── training/              # Training pipeline
│   ├── evaluation/            # Model evaluation
│   ├── serve/                 # API serving
│   └── monitoring/            # Model monitoring
├── configs/                   # Configuration files
├── tests/                     # Unit & integration tests
├── notebooks/                 # Exploratory analysis
├── scripts/                   # CLI scripts
└── docs/                      # Documentation
```

## Documentation Links

- [Project Report](docs/project_report.pdf) - Detailed report on methodology and results
- [EDA Notebook](notebooks/01_eda.ipynb) - Exploratory data analysis of credit card transactions
- [Feature Engineering Notebook](notebooks/02_feature_engineering.ipynb) - Feature creation and selection
- [Model Development Notebook](notebooks/03_model_development.ipynb) - Model training and evaluation
- [Monitoring Configuration](configs/monitoring.yaml) - Configuration for drift detection

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mlops_demo.git
cd mlops_demo

# Create virtual environment
python -m venv mlops_venv
source mlops_venv/bin/activate  # On Windows: mlops_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Download

```bash
# Download dataset using kagglehub
python scripts/download_data.py

# Or manually download from Kaggle and place in data/raw/
```

### 3. Run Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_eda.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_development.ipynb
```

### 4. Run Monitoring Script

```bash
# Execute monitoring script
python scripts/run_monitoring.py --config configs/monitoring.yaml
```

## Data Pipeline

- **Data Source**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Feature Engineering**: Time-based features, amount transformations, feature selection
- **Class Imbalance Handling**: SMOTE, SMOTE-Tomek, cost-sensitive learning

## Model Development

The project implements multiple models for fraud detection:

| Model | PR-AUC | ROC-AUC | F1 Score | Precision | Recall |
|-------|--------|---------|----------|-----------|--------|
| XGBoost (Tuned) | 0.9054 | 0.9967 | 0.7761 | 0.9231 | 0.6701 |
| Stacking Classifier | 0.9084 | 0.9962 | 0.7807 | 0.9149 | 0.6804 |
| Voting Classifier | 0.8609 | 0.9951 | 0.7342 | 0.8600 | 0.6392 |

For details on hyperparameters, feature importance, and model selection, see [Model Development Notebook](notebooks/03_model_development.ipynb).

## Monitoring and Drift Detection

The project includes a comprehensive monitoring system for:

- **Feature Drift Detection**: Using statistical tests to detect distribution shifts
- **Performance Monitoring**: Tracking precision, recall, and F1-score over time
- **Data Quality Checks**: Validating incoming data against expectations

Configuration for monitoring is available in [configs/monitoring.yaml](configs/monitoring.yaml).

## Testing

```bash
# Run tests
pytest tests/ -v
```

## Visualization Samples

The following visualizations are generated during analysis:

- Class distribution analysis ([view](images/class_distribution.png))
- Feature importance ([view](images/feature_importance.png))
- Time-based features ([view](images/time_features.png))
- Amount transformations ([view](images/amount_transformations.png))
- Model performance comparison ([view](images/model_comparison.png))

## Results

Our final model achieves:
- **PR-AUC**: 0.9084 (Stacking Classifier)
- **ROC-AUC**: 0.9962 (Stacking Classifier)
- **F1-Score**: 0.7807 (Stacking Classifier)
- **Precision**: 0.9149 (Stacking Classifier)
- **Recall**: 0.6804 (Stacking Classifier)

For complete results, see the [Project Report](docs/project_report.pdf).

## Future Work

- Integration with real-time streaming data
- Explainability enhancements for model decisions
- Active learning implementation for reducing false positives
- Enhanced visualization dashboard for monitoring

## References

1. Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Made With ML (MLOps Best Practices): https://madewithml.com/
3. Sahin, Y., & Duman, E. (2011). Detecting credit card fraud by ANN and logistic regression. 2011 International Symposium on Innovations in Intelligent Systems and Applications.
4. Dal Pozzolo, A., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Data Pipeline

Data flows through ingestion -> validation -> feature engineering -> training.

## Performance Benchmarks

See `docs/model_card.md` for latest metrics.
