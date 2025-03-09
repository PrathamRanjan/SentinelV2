"""
Batch process messages for scam detection
Process a CSV file with the scam detection API
"""
import os
import pandas as pd
import requests
import argparse
import time
import logging
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file(input_file, output_file, model_type="tfidf_xgboost", text_column="text", base_url="http://localhost:5000"):
    """Process a CSV file with the scam detection API"""
    try:
        # Check if API is available
        try:
            response = requests.get(f"{base_url}/api/health")
            if response.status_code != 200:
                logger.error(f"API is not available at {base_url}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to API: {str(e)}")
            return False
            
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        if text_column not in df.columns:
            logger.error(f"Error: Column '{text_column}' not found in the input file")
            logger.info(f"Available columns: {df.columns.tolist()}")
            return False
            
        total_rows = len(df)
        logger.info(f"Processing {total_rows} messages")
        
        results = []
        
        # Process each row
        for i, row in tqdm(df.iterrows(), total=total_rows, desc="Processing"):
            text = row[text_column]
            
            # Skip empty texts
            if pd.isna(text) or text.strip() == "":
                logger.warning(f"Skipping empty text at row {i}")
                continue
                
            # Call API
            try:
                response = requests.post(
                    f"{base_url}/api/detect",
                    json={"text": text, "model": model_type}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add result to row
                    result_row = row.to_dict()
                    result_row["is_scam"] = result["is_scam"]
                    result_row["confidence"] = result["confidence"]
                    result_row["risk_level"] = result["risk_level"]
                    result_row["indicators"] = ", ".join(result["scam_indicators"]) if result["scam_indicators"] else ""
                    
                    results.append(result_row)
                else:
                    logger.error(f"Error on row {i}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error processing row {i}: {str(e)}")
                
            # Small delay to prevent overwhelming the API
            time.sleep(0.01)
            
        # Create results dataframe and save
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Print summary
            scam_count = sum(results_df["is_scam"])
            logger.info(f"\nSummary:")
            logger.info(f"Total messages processed: {len(results_df)}")
            logger.info(f"Scam messages detected: {scam_count} ({scam_count/len(results_df)*100:.1f}%)")
            logger.info(f"Legitimate messages: {len(results_df) - scam_count} ({(len(results_df) - scam_count)/len(results_df)*100:.1f}%)")
            
            return True
        else:
            logger.error("No results obtained")
            return False
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process texts with scam detection API")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--column", type=str, default="text", help="Column containing text to analyze")
    parser.add_argument("--model", type=str, default="tfidf_xgboost", 
                       choices=["transformer", "tfidf_xgboost", "all", "best"],
                       help="Model to use for detection")
    parser.add_argument("--url", type=str, default="http://localhost:5000", help="Base URL of the API")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    success = process_file(args.input, args.output, args.model, args.column, args.url)
    
    if not success:
        logger.error("Batch processing failed")
        sys.exit(1)
    
    logger.info("Batch processing completed successfully")

if __name__ == "__main__":
    main()