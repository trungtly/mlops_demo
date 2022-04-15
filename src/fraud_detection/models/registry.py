"""
Model Registry for Fraud Detection System

Manages multiple model versions and provides model selection capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import joblib

from .schema import compatibility_layer, CompatibleModel

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    name: str
    version: str
    model_type: str
    file_path: str
    performance_metrics: Dict[str, float]
    feature_count: int
    training_date: str
    model_size_mb: float
    status: str = "active"  # active, deprecated, archived
    deployment_date: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Centralized model registry for fraud detection models"""
    
    def __init__(self, registry_path: str = "models/registry.json", models_dir: str = "models"):
        self.registry_path = Path(registry_path)
        self.models_dir = Path(models_dir)
        self.models: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, CompatibleModel] = {}
        
        # Load existing registry
        self._load_registry()
        
        # Auto-discover models if registry is empty
        if not self.models:
            self._discover_models()
    
    def _load_registry(self) -> None:
        """Load model registry from file"""
        if not self.registry_path.exists():
            logger.info("No existing registry found, will create new one")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            
            for model_id, model_data in data.items():
                self.models[model_id] = ModelMetadata(**model_data)
            
            logger.info(f"Loaded {len(self.models)} models from registry")
        
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save model registry to file"""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {model_id: asdict(metadata) for model_id, metadata in self.models.items()}
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved registry with {len(self.models)} models")
        
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _discover_models(self) -> None:
        """Auto-discover models in the models directory"""
        logger.info("Auto-discovering models...")
        
        model_files = list(self.models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            try:
                # Skip known auxiliary files
                if model_file.name in ["feature_scaler.pkl", "feature_selector.pkl"]:
                    continue
                
                # Try to load metadata
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    # Load from metadata file
                    metadata = self._load_model_metadata(metadata_file)
                else:
                    # Create basic metadata
                    metadata = self._create_basic_metadata(model_file)
                
                model_id = f"{metadata.name}_{metadata.version}"
                self.models[model_id] = metadata
                
                logger.info(f"Discovered model: {model_id}")
            
            except Exception as e:
                logger.warning(f"Failed to process {model_file}: {e}")
        
        # Save discovered models
        if self.models:
            self._save_registry()
    
    def _load_model_metadata(self, metadata_file: Path) -> ModelMetadata:
        """Load model metadata from JSON file"""
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        # Map common metadata formats to our structure
        return ModelMetadata(
            name=data.get("model_name", metadata_file.stem),
            version=data.get("version", "1.0"),
            model_type=data.get("model_type", "unknown"),
            file_path=str(metadata_file.with_suffix('.pkl')),
            performance_metrics=data.get("performance_metrics", {}),
            feature_count=data.get("feature_count", 0),
            training_date=data.get("creation_date", datetime.now().isoformat()),
            model_size_mb=self._get_file_size_mb(metadata_file.with_suffix('.pkl')),
            hyperparameters=data.get("hyperparameters", {})
        )
    
    def _create_basic_metadata(self, model_file: Path) -> ModelMetadata:
        """Create basic metadata for a model file"""
        return ModelMetadata(
            name=model_file.stem,
            version="1.0",
            model_type="unknown",
            file_path=str(model_file),
            performance_metrics={},
            feature_count=0,
            training_date=datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            model_size_mb=self._get_file_size_mb(model_file)
        )
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def register_model(self, 
                      name: str,
                      version: str,
                      model_path: str,
                      model_type: str,
                      performance_metrics: Dict[str, float],
                      feature_count: int,
                      hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model in the registry"""
        
        model_id = f"{name}_{version}"
        
        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists, updating...")
        
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model_type,
            file_path=model_path,
            performance_metrics=performance_metrics,
            feature_count=feature_count,
            training_date=datetime.now().isoformat(),
            model_size_mb=self._get_file_size_mb(Path(model_path)),
            hyperparameters=hyperparameters or {}
        )
        
        self.models[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def get_best_model(self, metric: str = "test_roc_auc") -> Optional[Tuple[str, ModelMetadata]]:
        """Get the best performing model based on a metric"""
        best_model_id = None
        best_score = -1
        
        for model_id, metadata in self.models.items():
            if metadata.status != "active":
                continue
            
            score = metadata.performance_metrics.get(metric, -1)
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        if best_model_id:
            return best_model_id, self.models[best_model_id]
        
        return None
    
    def list_models(self, status: Optional[str] = None) -> List[Tuple[str, ModelMetadata]]:
        """List all models, optionally filtered by status"""
        models = []
        for model_id, metadata in self.models.items():
            if status is None or metadata.status == status:
                models.append((model_id, metadata))
        
        return sorted(models, key=lambda x: x[1].training_date, reverse=True)
    
    def load_model(self, model_id: str) -> CompatibleModel:
        """Load a model with compatibility layer"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.models[model_id]
        
        try:
            # Use compatibility layer
            if compatibility_layer is not None:
                model = compatibility_layer.load_compatible_model(metadata.file_path)
            else:
                # Fallback loading
                model = joblib.load(metadata.file_path)
            
            self.loaded_models[model_id] = model
            logger.info(f"Loaded model: {model_id}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def compare_models(self, metric: str = "test_roc_auc") -> List[Tuple[str, float]]:
        """Compare all models based on a metric"""
        comparisons = []
        
        for model_id, metadata in self.models.items():
            if metadata.status == "active":
                score = metadata.performance_metrics.get(metric, 0)
                comparisons.append((model_id, score))
        
        return sorted(comparisons, key=lambda x: x[1], reverse=True)
    
    def deprecate_model(self, model_id: str) -> None:
        """Mark a model as deprecated"""
        if model_id in self.models:
            self.models[model_id].status = "deprecated"
            self._save_registry()
            
            # Remove from loaded models cache
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            
            logger.info(f"Deprecated model: {model_id}")
    
    def get_production_model(self) -> Optional[Tuple[str, CompatibleModel]]:
        """Get the current production model (best performing active model)"""
        best = self.get_best_model()
        if best:
            model_id, metadata = best
            model = self.load_model(model_id)
            return model_id, model
        return None
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry as dictionary"""
        return {model_id: asdict(metadata) for model_id, metadata in self.models.items()}


# Global registry instance
model_registry = ModelRegistry()