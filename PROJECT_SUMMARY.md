# MLOps Demo: Credit Card Fraud Detection

## Project Overview

This project demonstrates a comprehensive MLOps implementation for a credit card fraud detection system. It showcases best practices in machine learning operations, from data ingestion to model monitoring in production.

## Key Features

### Data Pipeline
- Robust data ingestion from Kaggle
- Data validation and preprocessing
- Train/validation/test splitting with stratification
- Synthetic data generation capabilities

### Feature Engineering
- Time-based features extraction
- Statistical features derived from transaction data
- Advanced feature interactions
- Outlier detection and handling

### Feature Selection
- Multiple selection strategies (variance-based, statistical tests, model-based)
- Correlation filtering to remove redundant features
- Ensemble selection combining multiple methods
- Stability selection via bootstrap sampling

### Model Training
- Support for multiple algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter optimization with grid search and Optuna
- Cross-validation with stratification for imbalanced data
- Ensemble model creation (voting and stacking)

### Model Evaluation
- Custom metrics for fraud detection (PR-AUC, ROC-AUC, recall at specific FPR thresholds)
- Cost-sensitive evaluation accounting for business impact
- Threshold optimization for different business scenarios
- Comprehensive performance reporting

### Model Monitoring
- Data drift detection between training and production data
- Performance monitoring with alerts on degradation
- Population Stability Index (PSI) calculation
- Advanced visualization for monitoring reports

### Model Serving
- FastAPI-based REST API
- Input validation and preprocessing
- Batch and single prediction endpoints
- Performance monitoring endpoints

## Development Timeline

This project was developed throughout 2021:

- **January 2021**: Initial project setup and configuration
- **March 2021**: Data ingestion and preprocessing implementation
- **May 2021**: Feature engineering pipeline development
- **June 2021**: Fixed data preprocessing scaling issues
- **July 2021**: Feature selection methods implementation
- **August 2021**: Performance optimization for model training
- **September 2021**: Model training pipeline with hyperparameter tuning
- **October 2021**: Custom evaluation metrics implementation
- **November 2021**: Model monitoring and drift detection
- **December 2021**: Model serving API and documentation

## Architecture

The project follows a modular architecture:

```
mlops_demo/
├── configs/             # Configuration files
├── data/                # Data storage
│   ├── raw/             # Raw data
│   ├── processed/       # Processed data
│   └── external/        # External data
├── src/                 # Source code
│   └── fraud_detection/
│       ├── data/        # Data processing modules
│       ├── features/    # Feature engineering
│       ├── models/      # Model definitions
│       ├── training/    # Training pipeline
│       ├── evaluation/  # Evaluation metrics
│       ├── monitoring/  # Drift detection & monitoring
│       └── serve/       # API serving
├── scripts/             # Utility scripts
├── notebooks/           # Analysis notebooks
├── tests/               # Unit and integration tests
├── artifacts/           # Model artifacts
└── model_registry/      # Versioned models
```

## Future Improvements

- Integration with MLflow for experiment tracking
- CI/CD pipeline for automated testing and deployment
- A/B testing framework for model deployment
- Containerization with Docker for easy deployment
- Online learning capabilities for continuous model updates
## Next Steps

- Implement A/B testing framework
- Add model versioning with DVC
