"""
Download and prepare SMS Spam Collection dataset for scam detection
Using kagglehub for simplified dataset access
"""
import os
import pandas as pd
import logging
import sys
import kagglehub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def download_sms_spam_dataset():
    """Download the SMS Spam Collection Dataset using kagglehub"""
    logger.info("Downloading SMS Spam Dataset")
    
    try:
        # Use kagglehub to download the dataset
        # This doesn't require explicit credential setup
        dataset_path = kagglehub.dataset_download("vishakhdapat/sms-spam-detection-dataset")
        
        logger.info(f"Dataset downloaded to: {dataset_path}")
        
        # Find CSV files in the downloaded path
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            logger.info(f"Found CSV files: {csv_files}")
            
            # Copy the first CSV file to our data directory for consistency
            import shutil
            target_path = os.path.join(DATA_DIR, 'spam_dataset_original.csv')
            shutil.copy(csv_files[0], target_path)
            
            logger.info(f"Copied dataset to: {target_path}")
            return target_path
        else:
            logger.warning("No CSV files found in the downloaded dataset")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return None

def prepare_sms_dataset(input_file, output_file=None):
    """Prepare the SMS Spam Collection dataset for training"""
    if output_file is None:
        output_file = os.path.join(DATA_DIR, 'scam_dataset.csv')
        
    logger.info(f"Preparing dataset from {input_file}")
    
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Display column names
        logger.info(f"Original column names: {df.columns.tolist()}")
        
        # Detect column format and standardize
        # Common formats:
        # 1. v1/v2 format (original UCI dataset)
        # 2. label/text format
        # 3. message/label format
        # 4. text/spam format
        
        # Case 1: v1/v2 format
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
        
        # Case 2: already in label/text format - do nothing
        
        # Case 3: message/label format
        elif 'message' in df.columns and any(col.lower() in ['label', 'spam', 'class'] for col in df.columns):
            label_col = next(col for col in df.columns if col.lower() in ['label', 'spam', 'class'])
            df = df[['message', label_col]]
            df.columns = ['text', 'label']
            
        # Case 4: text/spam format
        elif 'text' in df.columns and 'spam' in df.columns:
            df = df[['text', 'spam']]
            df.columns = ['text', 'label']
            
        # If we can't recognize the format, use the first two columns
        elif len(df.columns) >= 2:
            logger.warning(f"Unrecognized column format. Using first two columns: {df.columns[0]} and {df.columns[1]}")
            df = df.iloc[:, 0:2]
            df.columns = ['label', 'text']
        
        # Handle different label formats and convert to binary is_scam column
        if 'label' in df.columns:
            # Check label type
            if df['label'].dtype == 'object':
                # Handle text labels like 'spam'/'ham' or 'yes'/'no'
                unique_labels = df['label'].unique()
                logger.info(f"Unique label values: {unique_labels}")
                
                # Try to identify which value means "spam"
                spam_indicators = ['spam', 'yes', '1', 'true', 'scam']
                spam_value = None
                
                for value in unique_labels:
                    if str(value).lower() in spam_indicators:
                        spam_value = value
                        break
                
                if spam_value:
                    logger.info(f"Identified '{spam_value}' as the spam label")
                    df['is_scam'] = (df['label'] == spam_value).astype(int)
                else:
                    # If we can't determine, assume the first unique value is spam
                    logger.warning(f"Couldn't identify spam label. Assuming '{unique_labels[0]}' is spam")
                    df['is_scam'] = (df['label'] == unique_labels[0]).astype(int)
            else:
                # If it's already numeric
                df['is_scam'] = df['label'].astype(int)
                
            # Drop the original label column
            df.drop('label', axis=1, inplace=True)
        
        # Ensure text column exists
        if 'text' not in df.columns:
            # If no column named 'text', rename the remaining column
            remaining_col = df.columns[0] if df.columns[0] != 'is_scam' else df.columns[1]
            df.rename(columns={remaining_col: 'text'}, inplace=True)
        
        # Final dataset should have 'text' and 'is_scam' columns
        logger.info(f"Prepared dataset shape: {df.shape}")
        logger.info(f"Prepared column names: {df.columns.tolist()}")
        
        # Display class distribution
        logger.info(f"Class distribution:\n{df['is_scam'].value_counts()}")
        
        # Display sample rows
        logger.info("\nSample data:")
        logger.info(df.head(5))
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Prepared dataset saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}", exc_info=True)
        return None

def main():
    """Main function to download and prepare dataset"""
    try:
        # Download dataset
        input_file = download_sms_spam_dataset()
        
        if not input_file:
            logger.error("Failed to download dataset")
            sys.exit(1)
        
        # Prepare dataset
        output_file = os.path.join(DATA_DIR, 'scam_dataset.csv')
        prepared_file = prepare_sms_dataset(input_file, output_file)
        
        if not prepared_file:
            logger.error("Failed to prepare dataset")
            sys.exit(1)
        
        logger.info(f"Dataset ready for training: {prepared_file}")
        logger.info("You can now train your model using:")
        logger.info(f"curl -X POST http://localhost:5000/api/train -H 'Content-Type: application/json' -d '{{\"dataset_path\": \"{prepared_file}\"}}'")
        logger.info(f"Or using the test client:")
        logger.info(f"python test_client.py train --dataset {prepared_file} --model tfidf_xgboost")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()