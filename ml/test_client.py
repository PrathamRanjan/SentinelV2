"""
Test client for scam detection API
"""
import requests
import argparse
import json
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_health(base_url="http://localhost:5000"):
    """Check if the API is running and get model status"""
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            result = response.json()
            logger.info("API is running")
            
            # Check GPU info
            gpu_info = result.get('gpu_info', {})
            if gpu_info.get('available', False):
                logger.info(f"GPU available: {gpu_info.get('device_name')} (Count: {gpu_info.get('device_count')})")
            else:
                logger.info("No GPU available. Running on CPU.")
            
            # Check available models
            model_status = result.get('models', {})
            if model_status.get('transformer', False):
                logger.info("Transformer model is available")
            else:
                logger.warning("Transformer model not available")
                
            if model_status.get('tfidf_xgboost', False):
                logger.info("TF-IDF/XGBoost model is available")
            else:
                logger.warning("TF-IDF/XGBoost model not available")
                
            return result
        else:
            logger.error(f"API health check failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to API: {str(e)}")
        return None

def get_models_info(base_url="http://localhost:5000"):
    """Get detailed information about available models"""
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            result = response.json()
            
            available_models = result.get('available_models', [])
            logger.info(f"Available models: {', '.join(available_models) if available_models else 'None'}")
            
            # Display transformer model info
            if 'transformer' in result:
                transformer_info = result['transformer']
                logger.info("\nTransformer Model Info:")
                for key, value in transformer_info.items():
                    if key != 'available':
                        logger.info(f"  {key}: {value}")
            
            # Display TF-IDF/XGBoost model info
            if 'tfidf_xgboost' in result:
                tfidf_info = result['tfidf_xgboost']
                logger.info("\nTF-IDF/XGBoost Model Info:")
                for key, value in tfidf_info.items():
                    if key != 'available':
                        logger.info(f"  {key}: {value}")
            
            return result
        else:
            logger.error(f"Failed to get models info: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        return None

def detect_scam(text, model_type="all", base_url="http://localhost:5000"):
    """Test scam detection with the API"""
    try:
        logger.info(f"Detecting scam in text using model: {model_type}")
        
        response = requests.post(
            f"{base_url}/api/detect",
            json={"text": text, "model": model_type}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            logger.info("\n" + "="*60)
            logger.info(f"TEXT: {text}")
            logger.info("-"*60)
            
            # Check if there was an error
            if 'error' in result and result.get('is_scam') is None:
                logger.error(f"Error in API response: {result['error']}")
                return result
            
            # Display main result
            is_scam = result.get('is_scam', None)
            confidence = result.get('confidence', 0.0)
            risk_level = result.get('risk_level', 'unknown')
            
            if is_scam is not None:
                logger.info(f"RESULT: {'SCAM' if is_scam else 'NOT SCAM'}")
                logger.info(f"CONFIDENCE: {confidence:.4f}")
                logger.info(f"RISK LEVEL: {risk_level.upper()}")
            else:
                logger.warning("No prediction available")
            
            # Display scam indicators
            indicators = result.get('scam_indicators', [])
            if indicators:
                logger.info("\nSCAM INDICATORS:")
                for indicator in indicators:
                    logger.info(f"- {indicator}")
            else:
                logger.info("\nNo specific scam indicators detected")
            
            # Display individual model results if available
            if 'model_results' in result:
                logger.info("\nINDIVIDUAL MODEL RESULTS:")
                model_results = result['model_results']
                
                if 'transformer' in model_results:
                    tr_result = model_results['transformer']
                    tr_prediction = "SCAM" if tr_result.get('is_scam', False) else "NOT SCAM"
                    tr_confidence = tr_result.get('confidence', 0.0)
                    logger.info(f"Transformer: {tr_prediction} (Confidence: {tr_confidence:.4f})")
                
                if 'tfidf_xgboost' in model_results:
                    xgb_result = model_results['tfidf_xgboost']
                    xgb_prediction = "SCAM" if xgb_result.get('is_scam', False) else "NOT SCAM"
                    xgb_confidence = xgb_result.get('confidence', 0.0)
                    logger.info(f"TF-IDF/XGBoost: {xgb_prediction} (Confidence: {xgb_confidence:.4f})")
            
            logger.info("="*60)
            
            return result
        else:
            logger.error(f"API request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None

def train_models(dataset_path, model_type="all", transformer_model="roberta-base", epochs=3, base_url="http://localhost:5000"):
    """Train models using the API"""
    try:
        logger.info(f"Training {model_type} model(s) on dataset: {dataset_path}")
        
        # Prepare request data
        request_data = {
            "dataset_path": dataset_path,
            "model_type": model_type,
            "transformer_model": transformer_model,
            "epochs": epochs
        }
        
        # Send training request
        response = requests.post(
            f"{base_url}/api/train",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Training started: {result}")
            
            # Training will continue on the server, but we'll get an immediate response
            logger.info("The training process is running on the server.")
            logger.info("You can check the server logs for progress updates.")
            
            return result
        else:
            logger.error(f"Training request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error in training request: {str(e)}")
        return None

def test_with_examples(model_type="all", base_url="http://localhost:5000"):
    """Test the API with predefined examples"""
    examples = [
        "Congratulations! You've won a $1000 Amazon gift card. Click here to claim your prize: http://bit.ly/claim-prize",
        "URGENT: Your account has been compromised. Please verify your identity by providing your account details immediately.",
        "Hi Mom, just checking in to see how you're doing. Love you!",
        "Your package has been delayed. Track your delivery here: tracking.amazon.com/your-package",
        "Investment opportunity: Guaranteed 50% returns in just 3 months! Limited time offer.",
        "Meeting scheduled for tomorrow at 10am in the conference room. Please bring your project updates.",
        "We detected suspicious activity on your account. Please call us immediately at this number to verify your identity.",
        "Your iCloud account has been locked due to too many failed login attempts. Click here to unlock: security-icloud.com",
        "This is a reminder that your credit card payment is due on 15th of this month.",
        "BREAKING NEWS: Send this message to 10 friends and get free mobile recharge!"
    ]
    
    results = []
    for example in examples:
        result = detect_scam(example, model_type, base_url)
        results.append(result)
        print("\n")  # Add extra newline between examples
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test client for scam detection API")
    
    # API connection
    parser.add_argument("--url", type=str, default="http://localhost:5000", help="Base URL of the API")
    
    # Actions
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Health check
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Models info
    models_parser = subparsers.add_parser("models", help="Get models information")
    
    # Detection
    detect_parser = subparsers.add_parser("detect", help="Detect scam in text")
    detect_parser.add_argument("--text", type=str, help="Text to analyze")
    detect_parser.add_argument("--file", type=str, help="File containing text to analyze")
    detect_parser.add_argument("--model", type=str, default="all", 
                              choices=["transformer", "tfidf_xgboost", "all", "best"],
                              help="Model to use for detection")
    
    # Examples
    examples_parser = subparsers.add_parser("examples", help="Test with predefined examples")
    examples_parser.add_argument("--model", type=str, default="all", 
                                choices=["transformer", "tfidf_xgboost", "all", "best"],
                                help="Model to use for detection")
    
    # Training
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    train_parser.add_argument("--model", type=str, default="all", 
                             choices=["transformer", "tfidf_xgboost", "all"],
                             help="Model type to train")
    train_parser.add_argument("--transformer", type=str, default="roberta-base", 
                             help="Transformer model name")
    train_parser.add_argument("--epochs", type=int, default=3, 
                             help="Number of epochs for transformer training")
    
    args = parser.parse_args()
    
    # Perform action based on arguments
    if args.action == "health":
        check_api_health(args.url)
    
    elif args.action == "models":
        get_models_info(args.url)
    
    elif args.action == "detect":
        if args.text:
            detect_scam(args.text, args.model, args.url)
        elif args.file:
            try:
                with open(args.file, 'r') as f:
                    text = f.read()
                detect_scam(text, args.model, args.url)
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
        else:
            logger.error("Either --text or --file must be provided")
            parser.print_help()
    
    elif args.action == "examples":
        test_with_examples(args.model, args.url)
    
    elif args.action == "train":
        train_models(args.dataset, args.model, args.transformer, args.epochs, args.url)
    
    else:
        logger.error("No action specified")
        parser.print_help()

if __name__ == "__main__":
    main()