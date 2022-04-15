.PHONY: help install download train serve evaluate test lint format clean docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install project dependencies"
	@echo "  download    - Download fraud detection dataset"
	@echo "  train       - Train a fraud detection model"
	@echo "  serve       - Serve trained model as API"
	@echo "  evaluate    - Evaluate trained model"
	@echo "  test        - Run tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"

# Installation
install:
	pip install -e .
	pip install -r requirements.txt

# Data download
download:
	python scripts/download_data.py --output-dir data/raw/ --validate

download-force:
	python scripts/download_data.py --output-dir data/raw/ --force-download --validate

# Model training
train:
	python scripts/train_model.py --model xgboost --feature-engineering baseline

train-advanced:
	python scripts/train_model.py --model xgboost --feature-engineering advanced --feature-selection ensemble --hyperparameter-tuning

train-rf:
	python scripts/train_model.py --model random_forest --feature-engineering baseline

train-lgb:
	python scripts/train_model.py --model lightgbm --feature-engineering baseline

train-ensemble:
	python scripts/train_model.py --model voting --feature-engineering advanced

train-nn:
	python scripts/train_model.py --model deep_nn --feature-engineering baseline

# Model serving
serve:
	python scripts/serve_model.py --host 0.0.0.0 --port 8000

serve-dev:
	python scripts/serve_model.py --host 127.0.0.1 --port 8000 --reload --log-level DEBUG

# Model evaluation
evaluate:
	python scripts/evaluate_model.py --data-path data/raw/creditcard.csv --generate-plots --save-predictions

evaluate-model:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make evaluate-model MODEL=path/to/model.pkl"; exit 1; fi
	python scripts/evaluate_model.py --model-path $(MODEL) --data-path data/raw/creditcard.csv --generate-plots

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=fraud_detection --cov-report=html --cov-report=term

test-integration:
	python -m pytest tests/integration/ -v

# Code quality
lint:
	python -m flake8 src/ scripts/ tests/
	python -m mypy src/ --ignore-missing-imports

format:
	python -m black src/ scripts/ tests/
	python -m isort src/ scripts/ tests/

format-check:
	python -m black src/ scripts/ tests/ --check
	python -m isort src/ scripts/ tests/ --check-only

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Docker
docker-build:
	docker build -t fraud-detection .

docker-run:
	docker run -p 8000:8000 fraud-detection

docker-run-dev:
	docker run -p 8000:8000 -v $(PWD):/app fraud-detection

# Docker Compose - Full MLOps Stack
docker-up:
	docker-compose up -d

docker-up-build:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-down-volumes:
	docker-compose down -v

docker-logs:
	docker-compose logs -f

docker-logs-api:
	docker-compose logs -f fraud-detection-api

docker-logs-monitoring:
	docker-compose logs -f prometheus grafana

docker-restart:
	docker-compose restart

docker-status:
	docker-compose ps

# Production MLOps Stack
prod-deploy: docker-up-build
	@echo "Production MLOps stack deployed!"
	@echo "API: http://localhost:8000"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Prometheus: http://localhost:9090"

prod-status:
	@echo "=== MLOps Stack Status ==="
	docker-compose ps
	@echo ""
	@echo "=== Health Checks ==="
	curl -s http://localhost:8000/health | jq . || echo "API not responding"
	curl -s http://localhost:9090/-/healthy || echo "Prometheus not responding"
	curl -s http://localhost:3000/api/health | jq . || echo "Grafana not responding"

prod-monitoring:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Development setup
setup-dev: install
	pre-commit install
	@echo "Development environment setup complete!"

# Full pipeline
pipeline: download train evaluate
	@echo "Full pipeline completed!"

# Quick start
quickstart: install download train-rf serve-dev
	@echo "Quick start completed! Model is serving at http://localhost:8000"

# Experiment with multiple models
experiment:
	make train-rf
	make train-lgb  
	make train
	make train-ensemble
	@echo "Multiple model training completed!"

# Production deployment
deploy-prep: test lint format-check
	@echo "Pre-deployment checks passed!"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "API documentation available at http://localhost:8000/docs when serving"

# Health check
health-check:
	curl -f http://localhost:8000/health || echo "Service not running"

# Performance test
perf-test:
	@echo "Running performance tests..."
	python -c "import requests; import time; start=time.time(); [requests.post('http://localhost:8000/predict', json={'features': [0]*30}) for _ in range(100)]; print(f'100 requests in {time.time()-start:.2f}s')"

# Data validation
validate-data:
	python -c "from fraud_detection.data.validation import validate_dataset; validate_dataset('data/raw/creditcard.csv')"

# Model comparison
compare-models:
	python -c "from fraud_detection.models.base import ModelRegistry; registry = ModelRegistry(); print(registry.compare_models())"