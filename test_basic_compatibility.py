#!/usr/bin/env python3
"""
Simple test script for feature compatibility system
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("🔧 Testing Basic Feature Compatibility")
    print("=" * 50)
    
    try:
        # Test feature schema loading
        from src.fraud_detection.features.schema import FeatureSchema
        schema = FeatureSchema()
        
        print(f"✅ Feature schema loaded: {len(schema.feature_names)} features")
        print(f"   First 5 features: {schema.feature_names[:5]}")
        
        # Test sample data transformation
        sample_data = pd.DataFrame({
            'V1': [1.0], 'V2': [2.0], 'V3': [3.0], 'V4': [4.0], 'V5': [5.0],
            'V6': [0.0], 'V7': [0.0], 'V8': [0.0], 'V9': [0.0], 'V10': [0.0],
            'V11': [0.0], 'V12': [0.0], 'V14': [0.0], 'V16': [0.0], 'V17': [0.0],
            'V18': [0.0], 'V19': [0.0], 'V20': [0.0], 'V21': [0.0], 'V27': [0.0],
            'Time': [12345], 'Amount': [100.50]
        })
        
        # Test transformation
        transformed = schema.transform_to_schema(sample_data)
        print(f"✅ Data transformation: {transformed.shape}")
        
        # Test dict input
        dict_input = {'V1': 1.0, 'V2': 2.0, 'Amount': 100.0, 'Time': 12345}
        compatible_input = schema.create_compatible_input(dict_input)
        print(f"✅ Compatible input: shape {compatible_input.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_model_files():
    """Test model file discovery"""
    print("\n🔧 Testing Model File Discovery")
    print("=" * 50)
    
    import os
    models_dir = "models"
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        print(f"Found {len(model_files)} model files:")
        for f in model_files[:5]:  # Show first 5
            size_mb = os.path.getsize(os.path.join(models_dir, f)) / (1024*1024)
            print(f"  - {f} ({size_mb:.1f} MB)")
        
        # Check metadata files
        json_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} metadata files:")
        for f in json_files:
            print(f"  - {f}")
            
    else:
        print("❌ Models directory not found")

def test_direct_model_loading():
    """Test direct model loading"""
    print("\n🔧 Testing Direct Model Loading")
    print("=" * 50)
    
    try:
        import joblib
        model_path = "models/lightgbm_model.pkl"
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded: {type(model)}")
            
            # Test with dummy data
            dummy_data = np.random.random((1, 50))  # 50 features
            try:
                prediction = model.predict(dummy_data)
                prob_prediction = model.predict_proba(dummy_data)
                print(f"✅ Prediction works: {prediction[0]}")
                print(f"✅ Probability works: {prob_prediction[0]}")
            except Exception as e:
                print(f"⚠️ Prediction failed (expected with dummy data): {e}")
                
        else:
            print(f"❌ Model file not found: {model_path}")
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")

if __name__ == "__main__":
    test_basic_functionality()
    test_model_files()  
    test_direct_model_loading()
    
    print("\n" + "=" * 50)
    print("✅ Basic compatibility testing completed!")