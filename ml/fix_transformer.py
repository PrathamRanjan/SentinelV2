"""
Fix for transformer model training
Addresses the 'too many values to unpack' error
"""
import os
import sys
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_transformer(
    dataset_path,
    output_dir="./models/transformer",
    model_name="roberta-base",
    max_length=128,  # Reduced from 256 to avoid issues
    batch_size=16,
    epochs=3
):
    """Train transformer model with improved error handling"""
    
    logger.info(f"Training transformer model: {model_name}")
    logger.info(f"Using dataset: {dataset_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded with {len(df)} examples")
        
        # Clean data - important to avoid errors
        df = df.dropna(subset=['text'])  # Remove rows with missing text
        df['text'] = df['text'].astype(str)  # Ensure text is string
        
        # Limit text length to avoid issues
        df['text'] = df['text'].apply(lambda x: x[:1000] if len(x) > 1000 else x)
        
        # Ensure label is properly formatted
        df['is_scam'] = df['is_scam'].astype(int)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenization function with explicit handling
        def tokenize_function(examples):
            # Ensure text is a string
            text = str(examples['text'])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None  # Important: don't return tensors here
            )
            # Add label
            encoded['labels'] = examples['is_scam']
            return encoded
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(
            df, test_size=0.1, stratify=df['is_scam'], random_state=42
        )
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        # Explicitly map with batched=False to avoid shape issues
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=False,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function, 
            batched=False,
            remove_columns=eval_dataset.column_names
        )
        
        # Set up model
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        # Use a simple data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        logger.info("Starting training")
        trainer.train()
        
        # Evaluate model
        logger.info("Evaluating model")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save model info
        import json
        import time
        model_info = {
            'model_name': model_name,
            'max_length': max_length,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'eval_loss': eval_results.get('eval_loss', 0),
            'dataset_size': len(df),
            'model_type': 'transformer'
        }
        
        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python fix_transformer.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    train_transformer(dataset_path)