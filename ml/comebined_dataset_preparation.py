"""
combined_dataset_preparation.py - Combines SMS spam and phishing datasets for improved scam detection
"""
import os
import pandas as pd
import kagglehub
import logging
import sys

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
        dataset_path = kagglehub.dataset_download("vishakhdapat/sms-spam-detection-dataset")
        
        logger.info(f"SMS dataset downloaded to: {dataset_path}")
        
        # Find CSV files in the downloaded path
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            logger.info(f"Found SMS CSV files: {csv_files}")
            
            # Copy the first CSV file to our data directory for consistency
            import shutil
            target_path = os.path.join(DATA_DIR, 'spam_dataset_original.csv')
            shutil.copy(csv_files[0], target_path)
            
            logger.info(f"Copied SMS dataset to: {target_path}")
            return target_path
        else:
            logger.warning("No CSV files found in the SMS dataset")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading SMS dataset: {str(e)}")
        return None

def download_phishing_dataset():
    """Download phishing dataset using kagglehub"""
    logger.info("Downloading phishing dataset")
    
    try:
        # Use kagglehub to download the dataset
        dataset_path = kagglehub.dataset_download("shashwatwork/phishing-dataset-for-machine-learning")
        
        logger.info(f"Phishing dataset downloaded to: {dataset_path}")
        
        # Find CSV files in the downloaded path
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            logger.info(f"Found phishing CSV files: {csv_files}")
            
            # Copy the first CSV file to our data directory
            import shutil
            target_path = os.path.join(DATA_DIR, 'phishing_dataset_original.csv')
            shutil.copy(csv_files[0], target_path)
            
            logger.info(f"Copied phishing dataset to: {target_path}")
            return target_path
        else:
            logger.warning("No CSV files found in the phishing dataset")
            return None
    except Exception as e:
        logger.error(f"Error downloading phishing dataset: {str(e)}")
        return None

def prepare_sms_dataset(input_file):
    """Prepare the SMS Spam Collection dataset for training"""
    if not input_file:
        return None
        
    output_file = os.path.join(DATA_DIR, 'sms_scam_dataset.csv')
    logger.info(f"Preparing SMS dataset from {input_file}")
    
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"Original SMS dataset shape: {df.shape}")
        
        # Display column names
        logger.info(f"Original SMS column names: {df.columns.tolist()}")
        
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
                logger.info(f"Unique SMS label values: {unique_labels}")
                
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
        
        # Final dataset should have 'text' and 'is_scam' columns
        logger.info(f"Prepared SMS dataset shape: {df.shape}")
        logger.info(f"Prepared SMS column names: {df.columns.tolist()}")
        logger.info(f"SMS class distribution:\n{df['is_scam'].value_counts()}")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Prepared SMS dataset saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error preparing SMS dataset: {str(e)}", exc_info=True)
        return None

def prepare_phishing_dataset(input_file):
    """Prepare the phishing dataset for training"""
    if not input_file:
        return None
        
    output_file = os.path.join(DATA_DIR, 'phishing_scam_dataset.csv')
    logger.info(f"Preparing phishing dataset from {input_file}")
    
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"Original phishing dataset shape: {df.shape}")
        
        # Display column names
        logger.info(f"Original phishing column names: {df.columns.tolist()}")
        
        # Extract text content
        if 'text' not in df.columns:
            # Find text columns based on common names
            text_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'message', 'email', 'content', 'body', 'data'])]
            
            if text_cols:
                text_col = text_cols[0]
                logger.info(f"Using '{text_col}' as text column")
                df['text'] = df[text_col]
            else:
                # If no obvious text column, look for the column with longest string values
                max_len_col = None
                max_avg_len = 0
                
                for col in df.columns:
                    if df[col].dtype == object:  # Text columns are usually object type
                        avg_len = df[col].astype(str).str.len().mean()
                        if avg_len > max_avg_len:
                            max_avg_len = avg_len
                            max_len_col = col
                
                if max_len_col:
                    logger.info(f"Selected '{max_len_col}' as text column based on text length")
                    df['text'] = df[max_len_col]
                else:
                    logger.error("Could not determine text column in phishing dataset")
                    return None
        
        # Extract label
        if 'is_scam' not in df.columns:
            # Find label columns based on common names
            label_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['label', 'class', 'phishing', 'spam', 'target', 'type'])]
            
            if label_cols:
                label_col = label_cols[0]
                logger.info(f"Using '{label_col}' as label column")
                
                # Check unique values
                unique_values = df[label_col].unique()
                logger.info(f"Unique values in '{label_col}': {unique_values}")
                
                # Convert to binary format
                if df[label_col].dtype == object:
                    phishing_indicators = ['phishing', 'phish', 'scam', 'spam', 'malicious', 'bad', 'yes', 'true', '1']
                    
                    # Find which value indicates phishing
                    phish_value = None
                    for val in unique_values:
                        if str(val).lower() in phishing_indicators:
                            phish_value = val
                            break
                    
                    if phish_value:
                        logger.info(f"Identified '{phish_value}' as the phishing label")
                        df['is_scam'] = (df[label_col] == phish_value).astype(int)
                    else:
                        logger.warning("Could not identify phishing label value. Assuming all are phishing.")
                        df['is_scam'] = 1
                else:
                    # If it's already numeric
                    # Check if 0/1 or 1/0 encoding
                    if set(unique_values) == {0, 1}:
                        # Check which is more common - usually phishing examples are less common
                        value_counts = df[label_col].value_counts()
                        if value_counts.index[0] == 0 and value_counts.index[1] == 1:
                            # If 0 is more common, assume 1 is phishing
                            df['is_scam'] = df[label_col]
                        else:
                            # If 1 is more common, assume 0 is phishing
                            df['is_scam'] = 1 - df[label_col]
                    else:
                        # Just convert to int
                        df['is_scam'] = df[label_col].astype(int)
            else:
                # If we can't find a label column, assume all examples are phishing
                logger.warning("No label column found. Assuming all examples are phishing.")
                df['is_scam'] = 1
        
        # Keep only needed columns
        df = df[['text', 'is_scam']]
        
        # Clean text column - remove nulls and empty strings
        df = df[df['text'].notna()]
        df = df[df['text'].astype(str).str.strip() != '']
        
        logger.info(f"Prepared phishing dataset shape: {df.shape}")
        logger.info(f"Phishing class distribution:\n{df['is_scam'].value_counts()}")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Prepared phishing dataset saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error preparing phishing dataset: {str(e)}", exc_info=True)
        return None

def combine_datasets(spam_file, phishing_file):
    """Combine spam and phishing datasets"""
    if not spam_file or not phishing_file:
        logger.error("Missing required datasets for combining")
        return None
        
    output_file = os.path.join(DATA_DIR, 'combined_scam_dataset.csv')
    logger.info(f"Combining datasets: {spam_file} and {phishing_file}")
    
    try:
        # Load datasets
        spam_df = pd.read_csv(spam_file)
        phishing_df = pd.read_csv(phishing_file)
        
        logger.info(f"SMS dataset shape: {spam_df.shape}")
        logger.info(f"Phishing dataset shape: {phishing_df.shape}")
        
        # Add source column for analysis
        spam_df['source'] = 'sms'
        phishing_df['source'] = 'phishing'
        
        # Combine datasets
        combined_df = pd.concat([spam_df, phishing_df], ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        # Remove duplicates
        combined_df.drop_duplicates(subset=['text'], inplace=True)
        logger.info(f"Combined dataset shape after removing duplicates: {combined_df.shape}")
        
        # Display class distribution
        logger.info(f"Class distribution in combined dataset:\n{combined_df['is_scam'].value_counts()}")
        logger.info(f"Source distribution in combined dataset:\n{combined_df['source'].value_counts()}")
        
        # Save with source column for analysis
        combined_df.to_csv(output_file.replace('.csv', '_with_source.csv'), index=False)
        
        # Remove source column for training
        combined_df = combined_df[['text', 'is_scam']]
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Combined dataset saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error combining datasets: {str(e)}", exc_info=True)
        return None

def main():
    """Main function"""
    try:
        # Download datasets
        sms_dataset = download_sms_spam_dataset()
        phishing_dataset = download_phishing_dataset()
        
        # Prepare datasets
        prepared_sms = prepare_sms_dataset(sms_dataset)
        prepared_phishing = prepare_phishing_dataset(phishing_dataset)
        
        # Combine datasets
        combined_dataset = combine_datasets(prepared_sms, prepared_phishing)
        
        if combined_dataset:
            logger.info(f"Successfully created combined dataset: {combined_dataset}")
            logger.info("\nTo train with the combined dataset, run:")
            logger.info(f"python test_client.py train --dataset {combined_dataset} --model tfidf_xgboost")
        else:
            logger.error("Failed to create combined dataset")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()