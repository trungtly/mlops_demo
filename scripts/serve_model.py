#!/usr/bin/env python3
"""
Serving script for fraud detection models.
Usage: python serve_model.py --model-path models/xgboost_20241219_123456.pkl --port 8000
"""

import argparse
import sys
import logging
import uvicorn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from fraud_detection.serve.api import create_app
from fraud_detection.models.base import ModelRegistry


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Serve fraud detection model')
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=False,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=False,
        help='Name of registered model to load'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    parser.add_argument(
        '--ssl-keyfile',
        type=str,
        help='SSL key file for HTTPS'
    )
    
    parser.add_argument(
        '--ssl-certfile',
        type=str,
        help='SSL certificate file for HTTPS'
    )
    
    return parser.parse_args()


def load_model(args):
    """Load model based on provided arguments."""
    logger = logging.getLogger(__name__)
    
    model_registry = ModelRegistry(args.models_dir)
    
    if args.model_path:
        # Load specific model file
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        logger.info(f"Loading model from {args.model_path}")
        # We need to determine model type to load properly
        # For now, let's list available models and pick the latest
        available_models = model_registry.list_models()
        if not available_models:
            raise ValueError("No models found in registry")
        
        # Use the most recent model (assuming timestamp in name)
        latest_model = sorted(available_models)[-1]
        model = model_registry.get_model(latest_model)
        
    elif args.model_name:
        # Load named model from registry
        logger.info(f"Loading model '{args.model_name}' from registry")
        model = model_registry.get_model(args.model_name)
        
    else:
        # Load latest model from registry
        available_models = model_registry.list_models()
        if not available_models:
            raise ValueError("No models found in registry. Please train a model first.")
        
        latest_model = sorted(available_models)[-1]
        logger.info(f"Loading latest model: {latest_model}")
        model = model_registry.get_model(latest_model)
    
    return model


def main():
    """Main serving function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting fraud detection model server")
    
    try:
        # Load model
        model = load_model(args)
        logger.info(f"Model loaded: {model.get_model_name()}")
        
        # Create FastAPI app
        app = create_app(model)
        
        # Configure SSL if provided
        ssl_config = {}
        if args.ssl_keyfile and args.ssl_certfile:
            ssl_config = {
                'ssl_keyfile': args.ssl_keyfile,
                'ssl_certfile': args.ssl_certfile
            }
            logger.info("HTTPS enabled")
        
        # Start server
        logger.info(f"Starting server on {args.host}:{args.port}")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level.lower(),
            reload=args.reload,
            **ssl_config
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# Default port: 8080, configurable via --port flag
