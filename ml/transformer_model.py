"""
Transformer-based scam detection model using HuggingFace transformers
Leverages pre-trained language models fine-tuned for scam classification
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransformerScamDetector:
    def __init__(
        self, 
        model_name="roberta-base",  # Default model
        max_length=256,            # Max sequence length
        device=None,               # Auto-detect device
        output_dir="./models/transformer"  # Output directory
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir
        
        # Determine device (GPU/CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = None
        self.model = None
    
    def load_tokenizer(self):
        """Load tokenizer for the specified model"""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer for {self.model_name}")
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        return self.tokenizer
    
    def tokenize_data(self, texts):
        """Tokenize input texts"""
        tokenizer = self.load_tokenizer()
        return tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, df, text_column, label_column):
        """Prepare dataset for training"""
        # Ensure labels are in correct format
        df = df.copy()
        df[label_column] = df[label_column].astype(int)
        
        # Split data
        train_df, eval_df = train_test_split(df, test_size=0.1, stratify=df[label_column], random_state=42)
        logger.info(f"Training on {len(train_df)} examples, evaluating on {len(eval_df)} examples")
        
        # Convert to Hugging Face Datasets
        def encode_batch(batch):
            """Tokenize a batch of texts"""
            tokenizer = self.load_tokenizer()
            encoded = tokenizer(
                batch[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            return {
                **encoded,
                "labels": np.array(batch[label_column])
            }
        
        # Convert to HF datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            lambda x: encode_batch({text_column: [x[text_column]], label_column: [x[label_column]]}),
            batched=False,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            lambda x: encode_batch({text_column: [x[text_column]], label_column: [x[label_column]]}),
            batched=False,
            remove_columns=eval_dataset.column_names
        )
        
        return train_dataset, eval_dataset
    
    def train(self, dataset_path, text_column='text', label_column='is_scam', epochs=3, batch_size=16):
        """Train the transformer model"""
        logger.info(f"Loading dataset from {dataset_path}")
        start_time = time.time()
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded with {len(df)} examples")
            
            # Check GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {gpu_name}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                logger.info(f"GPU Memory: Allocated {memory_allocated:.2f} MB, Reserved {memory_reserved:.2f} MB")
            
            # Load tokenizer and model
            logger.info(f"Loading model: {self.model_name}")
            tokenizer = self.load_tokenizer()
            
            # Initialize model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2  # Binary classification
            )
            
            # Prepare datasets
            logger.info("Preparing datasets")
            train_dataset, eval_dataset = self.prepare_dataset(df, text_column, label_column)
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                report_to="none"  # Disable wandb, etc.
            )
            
            # Define Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            # Train model
            logger.info("Starting training")
            train_start_time = time.time()
            trainer.train()
            train_duration = time.time() - train_start_time
            logger.info(f"Training completed in {train_duration:.2f} seconds")
            
            # Evaluate model
            logger.info("Evaluating model")
            eval_result = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_result}")
            
            # Test on some examples to get more detailed metrics
            logger.info("Running detailed evaluation")
            eval_df = pd.DataFrame(columns=['text', 'true_label', 'pred_label', 'confidence'])
            
            # Sample from the original dataset for interpretable evaluation
            test_df = df.sample(min(100, len(df)), random_state=42)
            
            for _, row in test_df.iterrows():
                text = row[text_column]
                label = row[label_column]
                
                # Inference on single example
                encoded = tokenizer(
                    text, 
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Inference
                self.model.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**encoded)
                
                # Process outputs
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]
                confidence = probabilities[0][prediction].cpu().numpy().item()
                
                # Add to results
                eval_df = pd.concat([eval_df, pd.DataFrame({
                    'text': [text[:100] + "..." if len(text) > 100 else text],
                    'true_label': [label],
                    'pred_label': [prediction],
                    'confidence': [confidence]
                })])
            
            # Calculate metrics
            y_true = eval_df['true_label'].values
            y_pred = eval_df['pred_label'].values
            
            # Generate report
            accuracy = accuracy_score(y_true, y_pred)
            class_report = classification_report(y_true, y_pred)
            
            logger.info(f"Detailed evaluation accuracy: {accuracy:.4f}")
            logger.info(f"Classification report:\n{class_report}")
            
            # Save model
            logger.info(f"Saving model to {self.output_dir}")
            trainer.save_model(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            
            # Save model info
            model_info = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'training_time_seconds': train_duration,
                'accuracy': float(accuracy),
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': 'transformer'
            }
            
            with open(os.path.join(self.output_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model trained and saved in {time.time() - start_time:.2f} seconds")
            return True
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            return False
    
    def load_from_disk(self, model_dir=None):
        """Load model from disk"""
        if model_dir is None:
            model_dir = self.output_dir
        
        try:
            logger.info(f"Loading transformer model from {model_dir}")
            
            # Load model info
            with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
                model_info = json.load(f)
                
            # Update instance variables
            self.model_name = model_info.get('model_name', self.model_name)
            self.max_length = model_info.get('max_length', self.max_length)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded with accuracy: {model_info.get('accuracy', 'unknown')}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def predict(self, text):
        """Predict if a text is a scam"""
        try:
            # Ensure model is loaded
            if self.model is None or self.tokenizer is None:
                success = self.load_from_disk()
                if not success:
                    raise ValueError("Failed to load model")
            
            # Encode text
            encoded = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**encoded)
            
            # Process outputs
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]
            confidence = probabilities[0][prediction].cpu().numpy().item()
            
            # Return prediction
            return {
                "is_scam": bool(prediction),
                "confidence": float(confidence),
                "risk_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            }
        
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            return {
                "error": f"Error making prediction: {str(e)}",
                "is_scam": None,
                "confidence": 0.0
            }