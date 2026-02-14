"""
Flask Web Application for House Price Prediction

This application provides a web interface and REST API for predicting
house prices using the trained ML model.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Model path - will be set during initialization
MODEL_PATH = None
model = None

# Feature columns expected by the model (exact columns from training)
FEATURE_COLUMNS = [
    "Order",
    "PID",
    "MS SubClass",
    "Lot Frontage",
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
]


def find_latest_model() -> Path:
    """Find the latest trained model in mlruns directory."""
    mlruns_dir = Path('mlruns')
    
    # Look for model.pkl files
    model_files = list(mlruns_dir.rglob('model.pkl'))
    
    if not model_files:
        raise FileNotFoundError("No trained model found in mlruns directory")
    
    # Get the most recently modified model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found model at: {latest_model}")
    
    return latest_model


def load_model():
    """Load the trained model."""
    global model, MODEL_PATH
    
    try:
        MODEL_PATH = find_latest_model()
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': str(MODEL_PATH) if MODEL_PATH else None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects JSON with house features.
    Returns predicted price.
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No input data provided',
                'success': False
            }), 400
        
        # Create DataFrame with all features
        # For simplicity, we'll use a subset of key features
        # and fill others with defaults
        input_df = create_input_dataframe(data)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert from log-scale to actual price (model was trained on log1p(SalePrice))
        # np.expm1(x) = exp(x) - 1
        predicted_price = float(np.expm1(prediction[0]))
        
        logger.info(f"Prediction made: ${predicted_price:,.2f}")
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': f"${predicted_price:,.2f}"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


def create_input_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create input DataFrame from user input.
    
    This function maps simplified user inputs to the subset of numeric features
    expected by the model (37 features + Order and PID).
    """
    # Default values for the features the model expects
    defaults = {
        "Order": 1,  # Dummy value
        "PID": 1,    # Dummy value
        "MS SubClass": 60,
        "Lot Frontage": 70.0,
        "Lot Area": 10000,
        "Overall Qual": 5,
        "Overall Cond": 5,
        "Year Built": 2000,
        "Year Remod/Add": 2000,
        "Mas Vnr Area": 0.0,
        "BsmtFin SF 1": 0,
        "BsmtFin SF 2": 0,
        "Bsmt Unf SF": 0,
        "Total Bsmt SF": 0,
        "1st Flr SF": 1000,
        "2nd Flr SF": 0,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 1500,
        "Bsmt Full Bath": 0,
        "Bsmt Half Bath": 0,
        "Full Bath": 2,
        "Half Bath": 0,
        "Bedroom AbvGr": 3,
        "Kitchen AbvGr": 1,
        "TotRms AbvGrd": 6,
        "Fireplaces": 0,
        "Garage Yr Blt": 2000,
        "Garage Cars": 2,
        "Garage Area": 400,
        "Wood Deck SF": 0,
        "Open Porch SF": 0,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,
        "Pool Area": 0,
        "Misc Val": 0,
        "Mo Sold": 6,
        "Yr Sold": 2026,
    }
    
    # Map user input keys (from form) to model feature names
    key_mapping = {
        'GrLivArea': 'Gr Liv Area',
        'LotArea': 'Lot Area',
        'OverallQual': 'Overall Qual',
        'YearBuilt': 'Year Built',
        'BedroomAbvGr': 'Bedroom AbvGr',
        'FullBath': 'Full Bath',
        'TotRmsAbvGrd': 'TotRms AbvGrd',
        'GarageCars': 'Garage Cars'
    }
    
    # Update with user-provided values (map keys first)
    for key, value in data.items():
        # Map the key if it's in our mapping, otherwise use as-is
        mapped_key = key_mapping.get(key, key)
        if mapped_key in defaults:
            defaults[mapped_key] = value
    
    # Create DataFrame with columns in the exact order expected by the model
    df = pd.DataFrame([defaults], columns=FEATURE_COLUMNS)
    
    return df


@app.route('/api/features', methods=['GET'])
def get_features():
    """Return the list of features the model expects."""
    return jsonify({
        'features': FEATURE_COLUMNS,
        'count': len(FEATURE_COLUMNS)
    })


if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
