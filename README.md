# Credit Card Fraud Detection - MLOps Demo

A comprehensive MLOps demonstration project for credit card fraud detection, showcasing production-ready machine learning practices including data validation, experiment tracking, model serving, monitoring, and CI/CD automation.

## 🎯 Project Overview

This project demonstrates end-to-end MLOps practices for building a production-grade fraud detection system:

- **Dataset**: Credit Card Fraud Detection from Kaggle (284,807 transactions, 0.172% fraud rate)
- **Challenge**: Highly imbalanced classification problem
- **Approach**: Cost-sensitive learning with proper evaluation metrics
- **Architecture**: Modular, testable, and scalable design

## 🏗️ Architecture

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

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repo-url>
cd mlops_demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Data Download

```bash
# Download dataset using kagglehub
python scripts/download_data.py

# Or manually download from Kaggle and place in data/raw/
```

### 3. Training Pipeline

```bash
# Run complete training pipeline
python scripts/train_model.py --config configs/training.yaml

# Or run individual steps
python -m fraud_detection.data.ingestion
python -m fraud_detection.training.train
```

### 4. Model Evaluation

```bash
# Evaluate model performance
python scripts/evaluate_model.py --model-path artifacts/models/best_model.pkl
```

### 5. Serve Model

```bash
# Start API server
python scripts/serve_model.py

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, ..., 0.3]}'
```

## 📊 Model Performance

Our production model achieves:
- **ROC AUC**: 0.999+
- **PR AUC**: 0.85+
- **Recall@1% FPR**: 0.92+
- **F1 Score**: 0.88+

*Optimized for high recall to minimize missed fraud cases while maintaining acceptable precision.*

## 🔧 Key Features

### Data Pipeline
- ✅ Robust data validation with Great Expectations
- ✅ Feature engineering for temporal patterns
- ✅ Proper train/validation/test splits
- ✅ Data versioning and lineage tracking

### Model Development
- ✅ Multiple algorithms (XGBoost, LightGBM, Neural Networks)
- ✅ Hyperparameter optimization with Optuna
- ✅ Cross-validation with stratified sampling
- ✅ Cost-sensitive learning approaches

### MLOps Infrastructure
- ✅ Experiment tracking with MLflow
- ✅ Model registry and versioning
- ✅ Automated testing (unit, integration, data)
- ✅ CI/CD with GitHub Actions
- ✅ Container-based deployment
- ✅ Model monitoring and drift detection

### Production Serving
- ✅ FastAPI-based REST API
- ✅ Input validation and preprocessing
- ✅ Batch prediction support
- ✅ Health checks and monitoring endpoints
- ✅ Configurable decision thresholds

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test types
pytest tests/unit/ -v --cov=fraud_detection
pytest tests/integration/ -v
pytest tests/data/ -v

# Generate coverage report
pytest tests/ --cov=fraud_detection --cov-report=html
```

## 📈 Experiment Tracking

We use MLflow for comprehensive experiment tracking:

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access at http://localhost:5000
```

Track:
- Model parameters and hyperparameters
- Training and validation metrics
- Model artifacts and preprocessing pipelines
- Data versions and feature importance
- Model performance over time

## 🚀 Deployment

### Local Development
```bash
# Build Docker image
docker build -t fraud-detection:latest .

# Run container
docker run -p 8000:8000 fraud-detection:latest
```

### Production Deployment
```bash
# Deploy with docker-compose
docker-compose up -d

# Or deploy to cloud (examples provided for AWS/GCP)
```

## 📊 Monitoring

The system includes comprehensive monitoring:

- **Data Drift Detection**: Statistical tests on input features
- **Model Performance**: Real-time metrics tracking
- **System Health**: API response times, error rates
- **Business Impact**: Fraud detection rates, false positive costs

```bash
# Run monitoring dashboard
python -m fraud_detection.monitoring.dashboard
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Documentation

- [Model Card](docs/model_card.md) - Model details and performance
- [Data Card](docs/data_card.md) - Dataset information and preprocessing
- [API Documentation](docs/api_docs.md) - REST API reference
- [Deployment Guide](docs/deployment.md) - Production deployment instructions

## 🔒 Security Considerations

- Input validation and sanitization
- Model artifact integrity verification
- Secure credential management
- API rate limiting and authentication
- Data privacy and compliance (PCI DSS considerations)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Made With ML](https://madewithml.com/) for MLOps best practices inspiration
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Open source ML/MLOps community

---

*This project demonstrates production-ready MLOps practices. For questions or suggestions, please open an issue.*