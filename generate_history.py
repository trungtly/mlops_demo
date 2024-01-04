#!/usr/bin/env python3
"""Generate realistic backdated commits for GitHub history demo."""

import random
import subprocess
import os
import datetime

REPO = "/Users/trung.ly/workspace/trungtly/mlops_demo"
os.chdir(REPO)

# Date range: Jan 1 2024 to Feb 17 2026
START = datetime.datetime(2024, 1, 1, 0, 0, 0)
END = datetime.datetime(2026, 2, 17, 23, 59, 59)

def random_date():
    delta = END - START
    rand_seconds = random.randint(0, int(delta.total_seconds()))
    dt = START + datetime.timedelta(seconds=rand_seconds)
    # Bias toward working hours (8am-10pm) but allow some off-hours
    if random.random() < 0.8:
        dt = dt.replace(hour=random.randint(8, 22))
    dt = dt.replace(minute=random.randint(0, 59), second=random.randint(0, 59))
    return dt.strftime("%Y-%m-%dT%H:%M:%S+1100")

def commit(msg, date_str):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    subprocess.run(["git", "add", "-A"], cwd=REPO, env=env)
    subprocess.run(["git", "commit", "-m", msg, "--allow-empty"], cwd=REPO, env=env)

def append_line(filepath, line):
    with open(os.path.join(REPO, filepath), "a") as f:
        f.write(line + "\n")

def prepend_comment(filepath, comment):
    full = os.path.join(REPO, filepath)
    with open(full, "r") as f:
        content = f.read()
    with open(full, "w") as f:
        f.write(comment + "\n" + content)

def replace_in_file(filepath, old, new):
    full = os.path.join(REPO, filepath)
    with open(full, "r") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new, 1)
        with open(full, "w") as f:
            f.write(content)
        return True
    return False


# Define a pool of realistic changes
changes = []

# --- Config tweaks ---
changes.append(("configs/training.yaml", lambda: append_line("configs/training.yaml", "# batch_size tuning note: tested 32, 64, 128"), "tune: adjust batch size training notes"))
changes.append(("configs/training.yaml", lambda: append_line("configs/training.yaml", "# early_stopping_patience: consider increasing for larger datasets"), "config: add early stopping notes"))
changes.append(("configs/model.yaml", lambda: append_line("configs/model.yaml", "# ensemble: consider stacking with gradient boosting"), "config: add ensemble strategy note"))
changes.append(("configs/model.yaml", lambda: append_line("configs/model.yaml", "# regularization: l2 penalty=0.01 showed best results"), "config: document regularization findings"))
changes.append(("configs/monitoring.yaml", lambda: append_line("configs/monitoring.yaml", "# drift_threshold: 0.05 for PSI, 0.1 for KS test"), "config: document drift threshold values"))
changes.append(("configs/data.yaml", lambda: append_line("configs/data.yaml", "# stratified_split: maintain class ratio in train/val/test"), "config: note stratified split strategy"))
changes.append(("configs/experiment.yaml", lambda: append_line("configs/experiment.yaml", "# experiment tracking: log all hyperparameters"), "config: add experiment tracking reminder"))
changes.append(("configs/serve.yaml", lambda: append_line("configs/serve.yaml", "# health_check_interval: 30s recommended for production"), "config: add health check interval note"))

# --- Documentation updates ---
changes.append(("README.md", lambda: append_line("README.md", "\n## Performance Benchmarks\n\nSee `docs/model_card.md` for latest metrics."), "docs: add performance benchmarks section to README"))
changes.append(("README.md", lambda: append_line("README.md", "\n## Data Pipeline\n\nData flows through ingestion -> validation -> feature engineering -> training."), "docs: document data pipeline flow"))
changes.append(("CHANGELOG.md", lambda: append_line("CHANGELOG.md", "\n### Unreleased\n- Improved feature selection pipeline\n- Updated drift detection thresholds"), "docs: update changelog with recent changes"))
changes.append(("docs/model_card.md", lambda: append_line("docs/model_card.md", "\n## Limitations\n\n- Model performance may degrade on transactions outside training distribution\n- Real-time latency depends on feature computation overhead"), "docs: add model limitations section"))
changes.append(("docs/monitoring_guide.md", lambda: append_line("docs/monitoring_guide.md", "\n## Alert Escalation\n\nCritical drift alerts should trigger model retraining pipeline."), "docs: add alert escalation procedure"))
changes.append(("docs/data_card.md", lambda: append_line("docs/data_card.md", "\n## Data Quality Checks\n\n- Null rate should be < 1% for critical features\n- Transaction amount outliers flagged above 99.9th percentile"), "docs: add data quality check criteria"))
changes.append(("docs/api_docs.md", lambda: append_line("docs/api_docs.md", "\n## Rate Limiting\n\nAPI endpoints are rate limited to 1000 requests/minute per client."), "docs: add rate limiting documentation"))
changes.append(("PROJECT_SUMMARY.md", lambda: append_line("PROJECT_SUMMARY.md", "\n## Next Steps\n\n- Implement A/B testing framework\n- Add model versioning with DVC"), "docs: add next steps to project summary"))

# --- Python source improvements ---
py_files_comments = [
    ("src/fraud_detection/config.py", "# Configuration management for fraud detection pipeline"),
    ("src/fraud_detection/utils.py", "# Utility functions for data processing and model evaluation"),
    ("src/fraud_detection/data/ingestion.py", "# Data ingestion module - handles raw data loading and initial parsing"),
    ("src/fraud_detection/data/preprocessing.py", "# Preprocessing pipeline - cleaning, normalization, encoding"),
    ("src/fraud_detection/data/validation.py", "# Data validation using schema checks and statistical tests"),
    ("src/fraud_detection/features/engineering.py", "# Feature engineering - domain-specific transformations"),
    ("src/fraud_detection/features/schema.py", "# Feature schema definitions and type constraints"),
    ("src/fraud_detection/features/selection.py", "# Feature selection using importance scores and correlation analysis"),
    ("src/fraud_detection/models/base.py", "# Base model interface for fraud detection models"),
    ("src/fraud_detection/models/ensemble.py", "# Ensemble methods combining multiple base models"),
    ("src/fraud_detection/models/neural_network.py", "# Neural network model for fraud detection"),
    ("src/fraud_detection/models/registry.py", "# Model registry for versioning and deployment tracking"),
    ("src/fraud_detection/monitoring/drift.py", "# Data and concept drift detection"),
    ("src/fraud_detection/monitoring/performance.py", "# Real-time model performance monitoring"),
    ("src/fraud_detection/serve/api.py", "# REST API endpoints for model serving"),
    ("src/fraud_detection/serve/predictor.py", "# Prediction service wrapping model inference"),
    ("src/fraud_detection/training/train.py", "# Training pipeline orchestration"),
    ("src/fraud_detection/training/tune.py", "# Hyperparameter tuning with Optuna/grid search"),
    ("src/fraud_detection/evaluation/evaluate.py", "# Model evaluation framework"),
    ("src/fraud_detection/evaluation/metrics.py", "# Custom metrics for fraud detection evaluation"),
]

commit_msgs_for_comments = [
    "refactor: add module docstring to {module}",
    "docs: improve module documentation in {module}",
    "style: add header comment to {module}",
    "refactor: clarify module purpose in {module}",
]

for filepath, comment in py_files_comments:
    module = filepath.split("/")[-1].replace(".py", "")
    msg = random.choice(commit_msgs_for_comments).format(module=module)
    changes.append((filepath, lambda fp=filepath, c=comment: append_line(fp, f"\n{c}"), msg))

# --- Test file updates ---
changes.append(("tests/conftest.py", lambda: append_line("tests/conftest.py", "\n# Shared fixtures for fraud detection test suite"), "test: add shared fixture documentation"))
changes.append(("tests/conftest.py", lambda: append_line("tests/conftest.py", "# TODO: add parameterized fixtures for edge cases"), "test: note parameterized fixture TODO"))

# --- Script improvements ---
changes.append(("scripts/train_model.py", lambda: append_line("scripts/train_model.py", "\n# Usage: python scripts/train_model.py --config configs/training.yaml"), "docs: add usage example to train script"))
changes.append(("scripts/evaluate_model.py", lambda: append_line("scripts/evaluate_model.py", "\n# Evaluation outputs saved to artifacts/"), "docs: document evaluation output location"))
changes.append(("scripts/serve_model.py", lambda: append_line("scripts/serve_model.py", "\n# Default port: 8080, configurable via --port flag"), "docs: add serve script port documentation"))
changes.append(("scripts/download_data.py", lambda: append_line("scripts/download_data.py", "\n# Supports kaggle and local file sources"), "docs: document supported data sources"))

# --- Makefile ---
changes.append(("Makefile", lambda: append_line("Makefile", "\n# Run full pipeline: make all"), "docs: add Makefile usage comment"))
changes.append(("Makefile", lambda: append_line("Makefile", "# Clean artifacts: make clean"), "docs: add clean target documentation"))

# --- Docker ---
changes.append(("Dockerfile", lambda: append_line("Dockerfile", "\n# Multi-stage build for smaller production image"), "docs: add Dockerfile build strategy note"))
changes.append(("docker-compose.yml", lambda: append_line("docker-compose.yml", "\n# docker-compose up --build to rebuild images"), "docs: add docker-compose rebuild note"))

# --- Gitignore ---
changes.append((".gitignore", lambda: append_line(".gitignore", "\n# Experiment outputs\nexperiments/*/outputs/"), "chore: ignore experiment output directories"))
changes.append((".gitignore", lambda: append_line(".gitignore", "# Temporary notebooks\n*.tmp.ipynb"), "chore: ignore temporary notebook files"))

# --- Requirements ---
changes.append(("requirements.txt", lambda: append_line("requirements.txt", "# pinned for reproducibility"), "chore: add pinning note to requirements"))

# --- pyproject.toml ---
changes.append(("pyproject.toml", lambda: append_line("pyproject.toml", "\n# See README.md for development setup instructions"), "docs: add setup reference to pyproject.toml"))

# --- Env ---
changes.append((".env.example", lambda: append_line(".env.example", "# MODEL_VERSION=v1.0.0"), "config: add model version to env example"))
changes.append((".env.example", lambda: append_line(".env.example", "# LOG_LEVEL=INFO"), "config: add log level to env example"))
changes.append((".env.example", lambda: append_line(".env.example", "# MLFLOW_TRACKING_URI=http://localhost:5000"), "config: add mlflow tracking uri to env example"))

# --- Run scripts ---
changes.append(("run_eda.py", lambda: append_line("run_eda.py", "\n# EDA pipeline: generates visualizations and statistical summaries"), "docs: add EDA pipeline description"))
changes.append(("run_feature_engineering.py", lambda: append_line("run_feature_engineering.py", "\n# Feature engineering: transforms raw features into model-ready format"), "docs: add feature engineering description"))
changes.append(("run_model_development.py", lambda: append_line("run_model_development.py", "\n# Model development: training, evaluation, and selection pipeline"), "docs: add model development pipeline description"))

# --- Monitoring ---
changes.append(("monitoring/", lambda: append_line("monitoring/__init__.py" if os.path.exists(os.path.join(REPO, "monitoring/__init__.py")) else "docs/monitoring_guide.md", "\n## Monitoring Stack\n\nPrometheus + Grafana for infrastructure, custom drift detection for ML."), "docs: describe monitoring stack"))

# --- More realistic development-style commits ---
changes.append(("src/fraud_detection/__init__.py", lambda: append_line("src/fraud_detection/__init__.py", '\n__version__ = "0.2.0"'), "bump: version to 0.2.0"))
changes.append(("src/fraud_detection/__init__.py", lambda: (
    replace_in_file("src/fraud_detection/__init__.py", '"0.2.0"', '"0.3.0"') or
    append_line("src/fraud_detection/__init__.py", '# version bump pending')
), "bump: version to 0.3.0"))
changes.append(("src/fraud_detection/__init__.py", lambda: (
    replace_in_file("src/fraud_detection/__init__.py", '"0.3.0"', '"0.4.0"') or
    append_line("src/fraud_detection/__init__.py", '# version bump pending')
), "bump: version to 0.4.0"))
changes.append(("src/fraud_detection/__init__.py", lambda: (
    replace_in_file("src/fraud_detection/__init__.py", '"0.4.0"', '"1.0.0"') or
    append_line("src/fraud_detection/__init__.py", '# version bump pending')
), "bump: release version 1.0.0"))

# Shuffle and pick 75
random.shuffle(changes)
selected = changes[:75]

# Generate sorted random dates
dates = sorted([random_date() for _ in range(len(selected))])

print(f"Generating {len(selected)} commits...")
for i, ((filepath, action, msg), date_str) in enumerate(zip(selected, dates)):
    print(f"  [{i+1}/{len(selected)}] {date_str[:10]} - {msg}")
    try:
        action()
        commit(msg, date_str)
    except Exception as e:
        print(f"    SKIP: {e}")

print("Done!")
