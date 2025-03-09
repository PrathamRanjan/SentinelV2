"""
Advanced Scam Detection API with Transformer Models
Provides both traditional ML and transformer-based models for scam detection
"""
from flask import Flask, request, jsonify
import os
import json
import logging
import pandas as pd
import re
import traceback
from flask_cors import CORS
from dotenv import load_dotenv
import torch
import pickle

# Import ML components
from transformer_model import TransformerScamDetector
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up base paths
MODEL_BASE_PATH = os.getenv('MODEL_BASE_PATH', os.path.join(os.path.dirname(__file__), 'models'))
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_BASE_PATH, 'transformer')
TFIDF_MODEL_PATH = os.path.join(MODEL_BASE_PATH, 'tfidf_xgboost')

# Ensure directories exist
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
os.makedirs(TRANSFORMER_MODEL_PATH, exist_ok=True)
os.makedirs(TFIDF_MODEL_PATH, exist_ok=True)

# Global model instances
transformer_detector = None
tfidf_vectorizer = None
xgboost_model = None

# Text preprocessing functions for TFIDF model
def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Initialize models
def init_transformer_model():
    """Initialize the transformer model"""
    global transformer_detector
    
    # Check if model directory exists and has model files
    if os.path.exists(os.path.join(TRANSFORMER_MODEL_PATH, 'model_info.json')):
        logger.info("Loading existing transformer model")
        
        # Get model name from info file
        try:
            with open(os.path.join(TRANSFORMER_MODEL_PATH, 'model_info.json'), 'r') as f:
                model_info = json.load(f)
                model_name = model_info.get('model_name', 'roberta-base')
        except:
            model_name = 'roberta-base'
            
        # Create and load detector
        transformer_detector = TransformerScamDetector(
            model_name=model_name,
            output_dir=TRANSFORMER_MODEL_PATH
        )
        
        success = transformer_detector.load_from_disk()
        if success:
            logger.info("Transformer model loaded successfully")
        else:
            transformer_detector = None
            logger.warning("Failed to load transformer model")
    else:
        logger.info("No transformer model found")
        transformer_detector = None

def init_tfidf_xgboost_model():
    """Initialize TF-IDF and XGBoost models"""
    global tfidf_vectorizer, xgboost_model
    
    # Check if model files exist
    vectorizer_path = os.path.join(TFIDF_MODEL_PATH, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(TFIDF_MODEL_PATH, 'xgboost_model.json')
    
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        logger.info("Loading TF-IDF vectorizer and XGBoost model")
        
        try:
            # Load vectorizer
            with open(vectorizer_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
                
            # Load XGBoost model
            xgboost_model = xgb.Booster()
            xgboost_model.load_model(model_path)
            
            logger.info("TF-IDF and XGBoost models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading TF-IDF/XGBoost models: {str(e)}")
            tfidf_vectorizer = None
            xgboost_model = None
            return False
    else:
        logger.info("No TF-IDF/XGBoost models found")
        return False

# Training functions
def train_transformer_model(dataset_path, text_column='text', label_column='is_scam', model_name='roberta-base', epochs=3):
    """Train a transformer model"""
    global transformer_detector
    
    logger.info(f"Training transformer model with {model_name}")
    
    # Initialize detector
    transformer_detector = TransformerScamDetector(
        model_name=model_name,
        output_dir=TRANSFORMER_MODEL_PATH
    )
    
    # Train model
    success = transformer_detector.train(
        dataset_path=dataset_path,
        text_column=text_column,
        label_column=label_column,
        epochs=epochs
    )
    
    return success

def train_tfidf_xgboost_model(dataset_path, text_column='text', label_column='is_scam'):
    """Train TF-IDF and XGBoost models"""
    global tfidf_vectorizer, xgboost_model
    
    logger.info("Training TF-IDF and XGBoost models")
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Preprocess text
        df['processed_text'] = df[text_column].apply(clean_text)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X = df['processed_text']
        y = df[label_column].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
        dtest = xgb.DMatrix(X_test_tfidf, label=y_test)
        
        # Calculate class weight
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
        scale_pos_weight = class_weights[1] / class_weights[0]
        
        # Set parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist'
        }
        
        # Train model
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        xgboost_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate model
        y_pred_prob = xgboost_model.predict(dtest)
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
        
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"XGBoost model accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")
        
        # Save models
        vectorizer_path = os.path.join(TFIDF_MODEL_PATH, 'tfidf_vectorizer.pkl')
        model_path = os.path.join(TFIDF_MODEL_PATH, 'xgboost_model.json')
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
            
        xgboost_model.save_model(model_path)
        
        # Save model info
        model_info = {
            'accuracy': float(accuracy),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_features': int(tfidf_vectorizer.max_features)
        }
        
        with open(os.path.join(TFIDF_MODEL_PATH, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("TF-IDF and XGBoost models trained and saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error training TF-IDF/XGBoost models: {str(e)}")
        traceback.print_exc()
        return False

# Scam pattern detection
def detect_scam_patterns(text):
    """Detect common patterns in scam messages"""
    text = text.lower()
    indicators = []
    
    # Common scam patterns
    if any(word in text for word in ["urgent", "emergency", "immediate action"]):
        indicators.append("Creates false urgency")
    
    if any(word in text for word in ["congratulations", "winner", "won", "prize", "lottery"]):
        indicators.append("Promises unexpected rewards or winnings")
    
    if re.search(r'bank\s+details|account\s+number|password|credit\s+card|ssn|social\s+security', text):
        indicators.append("Requests sensitive financial information")
    
    if re.search(r'click\s+here|follow\s+this\s+link|download|attachment', text):
        indicators.append("Contains suspicious links or download requests")
    
    if any(word in text for word in ["guarantee", "risk-free", "100%", "double your money"]):
        indicators.append("Makes unrealistic promises or guarantees")
    
    if re.search(r'nigerian|foreign\s+prince|inheritance|overseas', text):
        indicators.append("Classic foreign money transfer scheme")
    
    if re.search(r'limited\s+time|offer\s+expires|act\s+now|today\s+only', text):
        indicators.append("Creates artificial time pressure")
    
    if re.search(r'secret|confidential|private|between\s+us', text):
        indicators.append("Requests secrecy or confidentiality")
    
    if re.search(r'government|irs|tax|refund|stimulus', text):
        indicators.append("Impersonates government agencies")
    
    if re.search(r'investment\s+opportunity|stock\s+tip|crypto|bitcoin|forex', text):
        indicators.append("Questionable investment opportunity")
    
    return indicators

# Initialize models at startup
@app.before_first_request
def initialize_models():
    """Initialize models when the application starts"""
    init_transformer_model()
    init_tfidf_xgboost_model()

# API endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check GPU availability
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }
    
    # Check model status
    model_status = {
        "transformer": transformer_detector is not None,
        "tfidf_xgboost": tfidf_vectorizer is not None and xgboost_model is not None
    }
    
    return jsonify({
        "status": "ok",
        "gpu_info": gpu_info,
        "models": model_status
    })

@app.route('/api/train', methods=['POST'])
def train_endpoint():
    """Endpoint to train models"""
    data = request.json
    
    # Validate input
    if not data or 'dataset_path' not in data:
        return jsonify({
            "error": "Missing dataset_path in request"
        }), 400
    
    dataset_path = data['dataset_path']
    text_column = data.get('text_column', 'text')
    label_column = data.get('label_column', 'is_scam')
    
    # Determine which models to train
    model_type = data.get('model_type', 'all').lower()
    
    results = {}
    
    # Train transformer model
    if model_type in ['all', 'transformer']:
        model_name = data.get('transformer_model', 'roberta-base')
        epochs = int(data.get('epochs', 3))
        
        logger.info(f"Training transformer model: {model_name}")
        transformer_success = train_transformer_model(
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
            model_name=model_name,
            epochs=epochs
        )
        
        results['transformer'] = {
            "success": transformer_success,
            "model_name": model_name
        }
    
    # Train TF-IDF and XGBoost models
    if model_type in ['all', 'tfidf_xgboost']:
        logger.info("Training TF-IDF and XGBoost models")
        tfidf_success = train_tfidf_xgboost_model(
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column
        )
        
        results['tfidf_xgboost'] = {
            "success": tfidf_success
        }
    
    return jsonify({
        "message": "Training completed",
        "results": results
    })

@app.route('/api/detect', methods=['POST'])
def detect_scam():
    """Endpoint to detect if text is a scam"""
    data = request.json
    
    # Validate input
    if not data or 'text' not in data:
        return jsonify({
            "error": "Missing text field in request"
        }), 400
    
    text = data['text']
    
    # Determine which model to use
    model_preference = data.get('model', 'best').lower()
    
    # Check available models
    transformer_available = transformer_detector is not None
    tfidf_available = tfidf_vectorizer is not None and xgboost_model is not None
    
    if not transformer_available and not tfidf_available:
        return jsonify({
            "error": "No models available. Please train models first."
        }), 503
    
    # Detect common scam patterns
    scam_indicators = detect_scam_patterns(text)
    
    # Make predictions
    results = {}
    
    # Use transformer model if available and preferred
    if transformer_available and model_preference in ['transformer', 'best', 'all']:
        try:
            transformer_result = transformer_detector.predict(text)
            results['transformer'] = transformer_result
        except Exception as e:
            logger.error(f"Error with transformer prediction: {str(e)}")
            results['transformer'] = {
                "error": str(e),
                "is_scam": None,
                "confidence": 0.0
            }
    
    # Use TF-IDF/XGBoost model if available and preferred
    if tfidf_available and model_preference in ['tfidf_xgboost', 'traditional', 'all']:
        try:
            # Clean text
            cleaned_text = clean_text(text)
            
            # Vectorize
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])
            
            # Convert to DMatrix
            dtext = xgb.DMatrix(text_tfidf)
            
            # Predict
            confidence = float(xgboost_model.predict(dtext)[0])
            prediction = 1 if confidence > 0.5 else 0
            
            results['tfidf_xgboost'] = {
                "is_scam": bool(prediction),
                "confidence": confidence,
                "risk_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            }
        except Exception as e:
            logger.error(f"Error with TF-IDF/XGBoost prediction: {str(e)}")
            results['tfidf_xgboost'] = {
                "error": str(e),
                "is_scam": None,
                "confidence": 0.0
            }
    
    # Determine final result based on preference
    final_result = None
    
    if model_preference == 'best':
        # Prefer transformer if available
        if 'transformer' in results and results['transformer'].get('is_scam') is not None:
            final_result = results['transformer']
        elif 'tfidf_xgboost' in results:
            final_result = results['tfidf_xgboost']
    elif model_preference == 'transformer' and 'transformer' in results:
        final_result = results['transformer']
    elif model_preference in ['tfidf_xgboost', 'traditional'] and 'tfidf_xgboost' in results:
        final_result = results['tfidf_xgboost']
    elif model_preference == 'all':
        # Use ensemble approach - average predictions if both available
        if 'transformer' in results and 'tfidf_xgboost' in results:
            transformer_conf = results['transformer'].get('confidence', 0)
            tfidf_conf = results['tfidf_xgboost'].get('confidence', 0)
            
            # If one model failed, use the other
            if results['transformer'].get('is_scam') is None:
                final_result = results['tfidf_xgboost']
            elif results['tfidf_xgboost'].get('is_scam') is None:
                final_result = results['transformer']
            else:
                # Otherwise average the predictions
                avg_confidence = (transformer_conf + tfidf_conf) / 2
                # If models disagree, use the more confident one
                if results['transformer'].get('is_scam') != results['tfidf_xgboost'].get('is_scam'):
                    if transformer_conf > tfidf_conf:
                        is_scam = results['transformer'].get('is_scam')
                    else:
                        is_scam = results['tfidf_xgboost'].get('is_scam')
                else:
                    is_scam = results['transformer'].get('is_scam')
                
                final_result = {
                    "is_scam": is_scam,
                    "confidence": avg_confidence,
                    "risk_level": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low",
                    "ensemble": True
                }
    
    # If no final result determined, use whatever is available
    if final_result is None:
        if 'transformer' in results:
            final_result = results['transformer']
        elif 'tfidf_xgboost' in results:
            final_result = results['tfidf_xgboost']
        else:
            final_result = {
                "is_scam": None,
                "confidence": 0.0,
                "error": "No predictions available"
            }
    
    # Add scam indicators to final result
    final_result['scam_indicators'] = scam_indicators
    
    # Add all model results if requested
    if model_preference == 'all':
        final_result['model_results'] = results
    
    return jsonify(final_result)

@app.route('/api/models', methods=['GET'])
def get_models_info():
    """Get information about available models"""
    models_info = {
        "available_models": []
    }
    
    # Check transformer model
    if transformer_detector is not None:
        try:
            with open(os.path.join(TRANSFORMER_MODEL_PATH, 'model_info.json'), 'r') as f:
                transformer_info = json.load(f)
            
            models_info['available_models'].append("transformer")
            models_info['transformer'] = transformer_info
        except:
            models_info['available_models'].append("transformer")
            models_info['transformer'] = {"available": True}
    
    # Check TF-IDF/XGBoost model
    if tfidf_vectorizer is not None and xgboost_model is not None:
        try:
            with open(os.path.join(TFIDF_MODEL_PATH, 'model_info.json'), 'r') as f:
                tfidf_info = json.load(f)
            
            models_info['available_models'].append("tfidf_xgboost")
            models_info['tfidf_xgboost'] = tfidf_info
        except:
            models_info['available_models'].append("tfidf_xgboost")
            models_info['tfidf_xgboost'] = {"available": True}
    
    return jsonify(models_info)

# Main function
if __name__ == '__main__':
    logger.info("Starting Advanced Scam Detection API")
    
    # Initialize models
    try:
        init_transformer_model()
        init_tfidf_xgboost_model()
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
    
    # Log status
    if transformer_detector is not None:
        logger.info("Transformer model is loaded and ready")
    else:
        logger.warning("No transformer model available")
    
    if tfidf_vectorizer is not None and xgboost_model is not None:
        logger.info("TF-IDF/XGBoost models are loaded and ready")
    else:
        logger.warning("No TF-IDF/XGBoost models available")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)