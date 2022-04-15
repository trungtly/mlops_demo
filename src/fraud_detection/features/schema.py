"""
Feature Schema Management for Fraud Detection Models

This module provides standardized feature handling across all models
to ensure compatibility and consistency.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureSchema:
    """Manages standardized feature schema for fraud detection models"""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path or "data/feature_info.json"
        self.schema = self._load_schema()
        self._scaler = self._load_scaler()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load feature schema from JSON file"""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default schema if file not found
            return self._create_default_schema()
    
    def _create_default_schema(self) -> Dict[str, Any]:
        """Create default feature schema"""
        return {
            "selected_features": [
                "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                "V11", "V12", "V14", "V16", "V17", "V18", "V19", "V20", "V21", "V27",
                "Time_hour", "Time_hour_sin", "Amount_above_p25",
                "V_mean", "V_std", "V_min", "V_max", "V_range", "V_median", "V_skew", "V_kurt",
                "V_neg_count", "V_pos_count",
                "V1_V2_interaction", "V1_V3_interaction", "V2_V3_interaction",
                "V4_V11_interaction", "V12_V14_interaction", "V10_V12_interaction",
                "Amount_V14_interaction", "Time_Amount_ratio",
                "V1_squared", "V1_cubed", "V2_squared", "V3_squared", "V3_cubed",
                "V4_squared", "V4_cubed", "V14_squared", "V14_cubed"
            ],
            "n_selected_features": 50
        }
    
    def _load_scaler(self) -> Optional[StandardScaler]:
        """Load the feature scaler"""
        scaler_path = "models/feature_scaler.pkl"
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError, ImportError) as e:
            # Create a basic scaler from schema parameters if pickle fails
            if hasattr(self.schema, 'scaler_params') and self.schema.get('scaler_params'):
                try:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(self.schema['scaler_params']['mean'])
                    scaler.scale_ = np.array(self.schema['scaler_params']['scale'])
                    scaler.n_features_in_ = len(scaler.mean_)
                    return scaler
                except Exception:
                    pass
            return None
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of standardized feature names"""
        return self.schema.get("selected_features", [])
    
    @property
    def n_features(self) -> int:
        """Get number of standardized features"""
        return len(self.feature_names)
    
    def validate_features(self, data: Union[pd.DataFrame, np.ndarray, List]) -> bool:
        """Validate that data has correct features"""
        if isinstance(data, pd.DataFrame):
            return list(data.columns) == self.feature_names
        elif isinstance(data, (np.ndarray, list)):
            if isinstance(data, list):
                data = np.array(data)
            if len(data.shape) == 1:
                return len(data) == self.n_features
            else:
                return data.shape[1] == self.n_features
        return False
    
    def transform_to_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform input data to match the standardized schema"""
        # If data already matches schema, return as-is
        if list(data.columns) == self.feature_names:
            return data
        
        # Create output dataframe with correct features
        result = pd.DataFrame(index=data.index)
        
        # Copy existing features that match schema
        for feature in self.feature_names:
            if feature in data.columns:
                result[feature] = data[feature]
            else:
                # Fill missing features with default values
                if feature.startswith('V') and '_' not in feature:
                    # Original V features - fill with 0
                    result[feature] = 0.0
                elif 'interaction' in feature:
                    # Interaction features - compute if possible
                    result[feature] = self._compute_interaction(data, feature)
                elif 'squared' in feature or 'cubed' in feature:
                    # Polynomial features - compute if possible
                    result[feature] = self._compute_polynomial(data, feature)
                else:
                    # Other features - fill with appropriate defaults
                    result[feature] = self._get_default_value(feature)
        
        return result[self.feature_names]  # Ensure correct order
    
    def _compute_interaction(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Compute interaction features"""
        # Parse feature name to get base features
        if feature_name == "V1_V2_interaction":
            return data.get("V1", 0) * data.get("V2", 0)
        elif feature_name == "V1_V3_interaction":
            return data.get("V1", 0) * data.get("V3", 0)
        elif feature_name == "V2_V3_interaction":
            return data.get("V2", 0) * data.get("V3", 0)
        elif feature_name == "V4_V11_interaction":
            return data.get("V4", 0) * data.get("V11", 0)
        elif feature_name == "V12_V14_interaction":
            return data.get("V12", 0) * data.get("V14", 0)
        elif feature_name == "V10_V12_interaction":
            return data.get("V10", 0) * data.get("V12", 0)
        elif feature_name == "Amount_V14_interaction":
            return data.get("Amount", 0) * data.get("V14", 0)
        elif feature_name == "Time_Amount_ratio":
            amount = data.get("Amount", 1)
            time_val = data.get("Time", 0)
            return np.where(amount != 0, time_val / amount, 0)
        else:
            return pd.Series(0.0, index=data.index)
    
    def _compute_polynomial(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Compute polynomial features"""
        if feature_name == "V1_squared":
            return data.get("V1", 0) ** 2
        elif feature_name == "V1_cubed":
            return data.get("V1", 0) ** 3
        elif feature_name == "V2_squared":
            return data.get("V2", 0) ** 2
        elif feature_name == "V3_squared":
            return data.get("V3", 0) ** 2
        elif feature_name == "V3_cubed":
            return data.get("V3", 0) ** 3
        elif feature_name == "V4_squared":
            return data.get("V4", 0) ** 2
        elif feature_name == "V4_cubed":
            return data.get("V4", 0) ** 3
        elif feature_name == "V14_squared":
            return data.get("V14", 0) ** 2
        elif feature_name == "V14_cubed":
            return data.get("V14", 0) ** 3
        else:
            return pd.Series(0.0, index=data.index)
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get appropriate default value for a feature"""
        if 'Time_hour' in feature_name:
            return 12.0  # Noon as default
        elif 'Amount_above' in feature_name:
            return 0.0
        elif feature_name.startswith('V_'):
            # Statistical features
            if 'mean' in feature_name:
                return 0.0
            elif 'std' in feature_name:
                return 1.0
            elif 'count' in feature_name:
                return 14.0  # Average count of V features
            else:
                return 0.0
        else:
            return 0.0
    
    def scale_features(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale features using the loaded scaler"""
        if self._scaler is None:
            # Return unscaled data if no scaler available
            if isinstance(data, pd.DataFrame):
                return data.values
            return data
        
        if isinstance(data, pd.DataFrame):
            return self._scaler.transform(data.values)
        return self._scaler.transform(data)
    
    def create_compatible_input(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Create model-compatible input from raw transaction data"""
        # Convert dict to DataFrame for easier manipulation
        df = pd.DataFrame([raw_data])
        
        # Transform to schema
        schema_df = self.transform_to_schema(df)
        
        # Scale features
        scaled_features = self.scale_features(schema_df)
        
        return scaled_features[0]  # Return single row
    
    def save_schema(self, path: Optional[str] = None):
        """Save current schema to file"""
        save_path = path or self.schema_path
        with open(save_path, 'w') as f:
            json.dump(self.schema, f, indent=4)


class ModelCompatibilityLayer:
    """Ensures all models use the same feature schema"""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.feature_schema = FeatureSchema(schema_path)
        self.model_cache = {}
    
    def load_compatible_model(self, model_path: str) -> Any:
        """Load a model and ensure it's compatible with the feature schema"""
        if model_path in self.model_cache:
            return self.model_cache[model_path]
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Wrap model with compatibility layer
            compatible_model = CompatibleModel(model, self.feature_schema)
            self.model_cache[model_path] = compatible_model
            return compatible_model
        
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def predict_compatible(self, model_path: str, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with automatic feature compatibility"""
        model = self.load_compatible_model(model_path)
        return model.predict(data)


class CompatibleModel:
    """Wrapper for models to ensure feature compatibility"""
    
    def __init__(self, model: Any, feature_schema: FeatureSchema):
        self.model = model
        self.feature_schema = feature_schema
    
    def predict(self, data: Union[Dict, pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Make predictions with automatic feature transformation"""
        # Handle different input types
        if isinstance(data, dict):
            # Single transaction
            features = self.feature_schema.create_compatible_input(data)
            return self.model.predict([features])
        
        elif isinstance(data, pd.DataFrame):
            # DataFrame input
            compatible_data = self.feature_schema.transform_to_schema(data)
            scaled_data = self.feature_schema.scale_features(compatible_data)
            return self.model.predict(scaled_data)
        
        elif isinstance(data, (np.ndarray, list)):
            # Array input - assume it's already in correct format
            if isinstance(data, list):
                data = np.array(data)
            
            # Check if it needs reshaping
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # Validate shape
            if data.shape[1] != self.feature_schema.n_features:
                raise ValueError(f"Input has {data.shape[1]} features, expected {self.feature_schema.n_features}")
            
            return self.model.predict(data)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def predict_proba(self, data: Union[Dict, pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Make probability predictions with automatic feature transformation"""
        # Handle different input types (same logic as predict)
        if isinstance(data, dict):
            features = self.feature_schema.create_compatible_input(data)
            return self.model.predict_proba([features])
        
        elif isinstance(data, pd.DataFrame):
            compatible_data = self.feature_schema.transform_to_schema(data)
            scaled_data = self.feature_schema.scale_features(compatible_data)
            return self.model.predict_proba(scaled_data)
        
        elif isinstance(data, (np.ndarray, list)):
            if isinstance(data, list):
                data = np.array(data)
            
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            if data.shape[1] != self.feature_schema.n_features:
                raise ValueError(f"Input has {data.shape[1]} features, expected {self.feature_schema.n_features}")
            
            return self.model.predict_proba(data)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


# Global instance for easy access
feature_schema = FeatureSchema()
compatibility_layer = ModelCompatibilityLayer()