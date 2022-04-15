"""
Production MLOps Deployment System
=================================

Enterprise-grade FastAPI application for fraud detection service with
advanced monitoring, caching, and production-ready features.
"""

import logging
import time
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest

try:
    from ..config import config
    from ..features.schema import compatibility_layer, feature_schema
except ImportError:
    # Fallback config for standalone usage
    class Config:
        API_HOST = "0.0.0.0"
        API_PORT = 8000
        API_WORKERS = 1
        MODELS_DIR = Path("models")
    config = Config()
    
    # Import compatibility layer with fallback
    try:
        from fraud_detection.features.schema import compatibility_layer, feature_schema
    except ImportError:
        # Create basic compatibility layer if not available
        compatibility_layer = None
        feature_schema = None

logger = logging.getLogger(__name__)

# Prometheus metrics for production monitoring
PREDICTION_COUNTER = Counter('fraud_predictions_total', 'Total fraud predictions', ['model_version', 'prediction'])
PREDICTION_LATENCY = Histogram('fraud_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('fraud_model_accuracy', 'Current model accuracy')
DRIFT_SCORE = Gauge('fraud_data_drift_score', 'Data drift score')
ERROR_COUNTER = Counter('fraud_prediction_errors_total', 'Total prediction errors')


class TransactionFeatures(BaseModel):
    """Transaction features for fraud prediction."""
    
    # Time features
    Time: float = Field(..., description="Time in seconds from first transaction")
    
    # Amount
    Amount: float = Field(..., ge=0, description="Transaction amount")
    
    # PCA features V1-V28
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    
    @validator('Amount')
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v


class BatchTransactionFeatures(BaseModel):
    """Batch of transactions for fraud prediction."""
    transactions: List[TransactionFeatures]
    
    @validator('transactions')
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError('Transactions list cannot be empty')
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    transaction_id: Optional[str] = None
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
    confidence: str
    risk_score: float = Field(..., ge=0, le=100)
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch fraud prediction."""
    predictions: List[PredictionResponse]
    total_transactions: int
    total_processing_time_ms: float
    fraud_detected_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class FraudDetectionService:
    """Fraud detection service with model management."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 threshold: float = 0.5):
        """
        Initialize the fraud detection service.
        
        Args:
            model_path: Path to the trained model
            threshold: Decision threshold for fraud classification
        """
        self.model = None
        self.preprocessor = None
        self.threshold = threshold
        self.model_version = "unknown"
        self.start_time = time.time()
        self.prediction_count = 0
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model with feature compatibility."""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Use compatibility layer if available
            if compatibility_layer is not None:
                self.model = compatibility_layer.load_compatible_model(str(model_path))
                logger.info(f"Model loaded with compatibility layer: {model_path}")
            else:
                # Fallback to direct loading
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded directly: {model_path}")
            
            self.model_version = model_path.stem  # Use filename as version
            logger.info(f"Model version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_features(self, features: TransactionFeatures) -> np.ndarray:
        """Preprocess transaction features."""
        # Convert to pandas DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Apply same feature engineering as in training
        # Time features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Is_Night'] = ((df['Hour'] >= 23) | (df['Hour'] <= 5)).astype(int)
        df['Is_Weekend'] = 0  # Would need date information
        
        # Amount features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_zscore'] = np.abs((df['Amount'] - df['Amount'].mean()) / df['Amount'].std())
        df['Amount_quantile'] = 5  # Default middle quantile for single prediction
        
        # Select features (exclude Time as it's not used in final model)
        feature_cols = [col for col in df.columns if col != 'Time']
        
        return df[feature_cols].values
    
    def predict_single(self, features: TransactionFeatures) -> PredictionResponse:
        """Make prediction for a single transaction."""
        start_time = time.time()
        
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Use compatibility layer if available
            if compatibility_layer is not None:
                # Convert features to dict for compatibility layer
                feature_dict = features.dict()
                fraud_probabilities = self.model.predict_proba(feature_dict)
                fraud_probability = fraud_probabilities[0, 1] if fraud_probabilities.ndim > 1 else fraud_probabilities[0]
            else:
                # Fallback to traditional preprocessing
                X = self.preprocess_features(features)
                fraud_probability = self.model.predict_proba(X)[0, 1]
            
            is_fraud = fraud_probability >= self.threshold
            
            # Calculate confidence and risk score
            confidence = self._calculate_confidence(fraud_probability)
            risk_score = fraud_probability * 100
            
            processing_time = (time.time() - start_time) * 1000
            self.prediction_count += 1
            
            return PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=float(fraud_probability),
                confidence=confidence,
                risk_score=float(risk_score),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, batch: BatchTransactionFeatures) -> BatchPredictionResponse:
        """Make predictions for a batch of transactions."""
        start_time = time.time()
        
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            predictions = []
            fraud_count = 0
            
            for i, transaction in enumerate(batch.transactions):
                pred_start = time.time()
                
                # Preprocess and predict
                X = self.preprocess_features(transaction)
                fraud_probability = self.model.predict_proba(X)[0, 1]
                is_fraud = fraud_probability >= self.threshold
                
                if is_fraud:
                    fraud_count += 1
                
                confidence = self._calculate_confidence(fraud_probability)
                risk_score = fraud_probability * 100
                pred_time = (time.time() - pred_start) * 1000
                
                predictions.append(PredictionResponse(
                    transaction_id=f"batch_{i}",
                    is_fraud=is_fraud,
                    fraud_probability=float(fraud_probability),
                    confidence=confidence,
                    risk_score=float(risk_score),
                    processing_time_ms=pred_time
                ))
            
            total_time = (time.time() - start_time) * 1000
            self.prediction_count += len(batch.transactions)
            
            return BatchPredictionResponse(
                predictions=predictions,
                total_transactions=len(batch.transactions),
                total_processing_time_ms=total_time,
                fraud_detected_count=fraud_count
            )
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability >= 0.8 or probability <= 0.2:
            return "high"
        elif probability >= 0.6 or probability <= 0.4:
            return "medium"
        else:
            return "low"
    
    def get_health(self) -> HealthResponse:
        """Get service health status."""
        uptime = time.time() - self.start_time
        
        return HealthResponse(
            status="healthy" if self.model is not None else "unhealthy",
            model_loaded=self.model is not None,
            model_version=self.model_version,
            uptime_seconds=uptime
        )


# Initialize service
service = FraudDetectionService()

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection service",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting Fraud Detection API")
    
    # Try to load model from default location
    default_model = config.MODELS_DIR / "latest" / "model.pkl"
    if default_model.exists():
        service.load_model(str(default_model))


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return service.get_health()


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionFeatures):
    """Predict fraud for a single transaction."""
    return service.predict_single(transaction)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(batch: BatchTransactionFeatures):
    """Predict fraud for a batch of transactions."""
    return service.predict_batch(batch)


@app.post("/model/load")
async def load_model(model_path: str):
    """Load a new model."""
    try:
        service.load_model(model_path)
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    health = service.get_health()
    
    return {
        "uptime_seconds": health.uptime_seconds,
        "total_predictions": service.prediction_count,
        "model_version": service.model_version,
        "predictions_per_second": service.prediction_count / health.uptime_seconds if health.uptime_seconds > 0 else 0
    }


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fraud detection API")
    parser.add_argument("--host", type=str, default=config.API_HOST, help="Host address")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port number")
    parser.add_argument("--workers", type=int, default=config.API_WORKERS, help="Number of workers")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fraud detection threshold")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load model if provided
    if args.model_path:
        service.load_model(args.model_path)
        service.threshold = args.threshold
    
    # Run server
    uvicorn.run(
        "fraud_detection.serve.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )


if __name__ == "__main__":
    main()