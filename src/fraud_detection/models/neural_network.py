from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from .base import BaseFraudModel


class MLPFraudModel(BaseFraudModel):
    """Multi-Layer Perceptron using sklearn."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _create_model(self) -> BaseEstimator:
        return MLPClassifier(**self.config)
    
    def get_model_name(self) -> str:
        return "MLP"


class KerasNeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make Keras model compatible with sklearn interface."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: list = [128, 64, 32],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 class_weight: Optional[Dict] = None,
                 random_state: int = 42):
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self):
        """Build the neural network architecture."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.hidden_layers[0], 
            input_dim=self.input_dim,
            activation='relu',
            kernel_initializer='he_normal'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(
                units, 
                activation='relu',
                kernel_initializer='he_normal'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return model
    
    def fit(self, X, y, **kwargs):
        """Fit the neural network."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self._build_model()
        
        # Set up callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            class_weight=self.class_weight,
            verbose=0
        )
        
        self.history = history
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make binary predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return (predictions.flatten() >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        prob_positive = self.model.predict(X_scaled, verbose=0).flatten()
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def get_params(self, deep=True):
        """Get model parameters."""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DeepNeuralNetworkFraudModel(BaseFraudModel):
    """Deep Neural Network model for fraud detection using Keras."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'random_state': 42
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.input_dim = None
    
    def _create_model(self) -> BaseEstimator:
        if self.input_dim is None:
            raise ValueError("input_dim must be set before creating model")
        
        # Calculate class weights for imbalanced data
        class_weight = {0: 1.0, 1: 100.0}  # Adjust based on class imbalance
        
        return KerasNeuralNetworkWrapper(
            input_dim=self.input_dim,
            hidden_layers=self.config['hidden_layers'],
            dropout_rate=self.config['dropout_rate'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            early_stopping_patience=self.config['early_stopping_patience'],
            class_weight=class_weight,
            random_state=self.config['random_state']
        )
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'DeepNeuralNetworkFraudModel':
        """Fit the model with input dimension handling."""
        
        self.input_dim = X.shape[1]
        
        return super().fit(X, y, X_val, y_val, **kwargs)
    
    def get_model_name(self) -> str:
        return "DeepNeuralNetwork"
    
    def get_training_history(self) -> Optional[Dict]:
        """Get training history if available."""
        if self.is_fitted and hasattr(self.model, 'history'):
            return self.model.history.history
        return None


class AutoencoderFraudModel(BaseFraudModel):
    """Autoencoder-based anomaly detection for fraud."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'encoding_layers': [64, 32, 16],
            'decoding_layers': [16, 32, 64],
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'validation_split': 0.2,
            'contamination': 0.002,  # Expected fraud rate
            'random_state': 42
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.input_dim = None
        self.threshold = None
    
    def _build_autoencoder(self):
        """Build autoencoder architecture."""
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.config['encoding_layers']:
            encoded = layers.Dense(units, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
        
        # Decoder
        decoded = encoded
        for units in self.config['decoding_layers']:
            decoded = layers.Dense(units, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse'
        )
        
        return autoencoder
    
    def _create_model(self):
        """Create autoencoder wrapper."""
        return AutoencoderWrapper(
            input_dim=self.input_dim,
            encoding_layers=self.config['encoding_layers'],
            decoding_layers=self.config['decoding_layers'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            contamination=self.config['contamination'],
            random_state=self.config['random_state']
        )
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'AutoencoderFraudModel':
        """Fit autoencoder on normal transactions only."""
        
        self.input_dim = X.shape[1]
        
        # Train only on normal transactions (class 0)
        X_normal = X[y == 0]
        
        return super().fit(X_normal, pd.Series([0] * len(X_normal)), **kwargs)
    
    def get_model_name(self) -> str:
        return "Autoencoder"


class AutoencoderWrapper(BaseEstimator, ClassifierMixin):
    """Autoencoder wrapper for anomaly detection."""
    
    def __init__(self, 
                 input_dim: int,
                 encoding_layers: list = [64, 32, 16],
                 decoding_layers: list = [16, 32, 64],
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 validation_split: float = 0.2,
                 contamination: float = 0.002,
                 random_state: int = 42):
        
        self.input_dim = input_dim
        self.encoding_layers = encoding_layers
        self.decoding_layers = decoding_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.contamination = contamination
        self.random_state = random_state
        
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _build_autoencoder(self):
        """Build autoencoder model."""
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.encoding_layers:
            encoded = layers.Dense(units, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
        
        # Decoder
        decoded = encoded
        for units in self.decoding_layers:
            decoded = layers.Dense(units, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return autoencoder
    
    def fit(self, X, y=None):
        """Train autoencoder on normal data."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build autoencoder
        self.autoencoder = self._build_autoencoder()
        
        # Train autoencoder
        self.autoencoder.fit(
            X_scaled, X_scaled,  # Reconstruct input
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            verbose=0
        )
        
        # Calculate threshold based on reconstruction error
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Set threshold at the contamination percentile
        self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict anomalies based on reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Return 1 for anomalies (fraud), 0 for normal
        return (reconstruction_errors > self.threshold).astype(int)
    
    def predict_proba(self, X):
        """Return reconstruction error as anomaly score."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Normalize errors to [0, 1] range
        normalized_errors = reconstruction_errors / (self.threshold * 2)
        normalized_errors = np.clip(normalized_errors, 0, 1)
        
        prob_normal = 1 - normalized_errors
        prob_anomaly = normalized_errors
        
        return np.column_stack([prob_normal, prob_anomaly])


def create_neural_network_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseFraudModel:
    """Factory function to create neural network models."""
    
    models = {
        'mlp': MLPFraudModel,
        'deep_nn': DeepNeuralNetworkFraudModel,
        'autoencoder': AutoencoderFraudModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](config)
# Neural network model for fraud detection
