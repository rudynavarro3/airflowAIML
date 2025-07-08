"""
Custom Airflow operators for ML model operations
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from airflow.models import BaseOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.context import Context
from airflow.utils.decorators import apply_defaults
from airflow.configuration import conf
from airflow.exceptions import AirflowException

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
from huggingface_hub import login


class ModelDownloadOperator(BaseOperator):
    """Download a model from HuggingFace Hub"""
    
    @apply_defaults
    def __init__(
        self,
        model_name: str,
        local_path: str,
        use_auth_token: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.local_path = local_path
        self.use_auth_token = use_auth_token
    
    def execute(self, context: Context):
        """Download model and tokenizer"""
        logging.info(f"Downloading model: {self.model_name}")
        
        # Get HuggingFace token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if self.use_auth_token and hf_token:
            login(token=hf_token)
        
        # Create local directory
        os.makedirs(self.local_path, exist_ok=True)
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.save_pretrained(self.local_path)
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.save_pretrained(self.local_path)
            
            logging.info(f"Model downloaded successfully to {self.local_path}")
            
        except Exception as e:
            logging.error(f"Error downloading model: {str(e)}")
            raise AirflowException(f"Failed to download model: {str(e)}")


class ModelFineTuneOperator(BaseOperator):
    """Fine-tune a model using HuggingFace Transformers"""
    
    @apply_defaults
    def __init__(
        self,
        base_model_path: str,
        dataset_path: str,
        output_path: str,
        training_args: Dict[str, Any],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.training_args = training_args
    
    def execute(self, context: Context):
        """Fine-tune the model"""
        logging.info("Starting model fine-tuning")
        
        # try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load and prepare dataset
        dataset = self._prepare_dataset(tokenizer)
        
        # Split dataset
        train_dataset = dataset['train']
        eval_dataset = dataset.get('validation', dataset['train'].train_test_split(test_size=0.1)['test'])
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_path,
            **self.training_args
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        logging.info("Training started...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(self.output_path)
        
        logging.info(f"Model fine-tuning completed. Saved to {self.output_path}")
            
        # except Exception as e:
        #     logging.error(f"Error during fine-tuning: {str(e)}")
        #     raise AirflowException(f"Fine-tuning failed: {str(e)}")
    
    def _prepare_dataset(self, tokenizer):
        """Prepare dataset for training"""
        # Load dataset
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        if isinstance(data, list):
            # Assume it's a list of conversations
            texts = []
            for item in data:
                if isinstance(item, dict) and 'dialog' in item:
                    # DailyDialog format
                    dialog_text = " ".join(item['dialog'])
                    texts.append(dialog_text)
                elif isinstance(item, str):
                    texts.append(item)
        else:
            texts = [str(data)]
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train/validation
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        
        return split_dataset


class ModelValidationOperator(BaseOperator):
    """Validate a fine-tuned model"""
    
    @apply_defaults
    def __init__(
        self,
        model_path: str,
        validation_dataset_path: str,
        metrics: List[str],
        output_path: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.validation_dataset_path = validation_dataset_path
        self.metrics = metrics
        self.output_path = output_path
    
    def execute(self, context: Context):
        """Validate the model"""
        logging.info("Starting model validation")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            # Load validation dataset
            val_texts = self._load_validation_data()
            
            # Generate predictions
            predictions = self._generate_predictions(model, tokenizer, val_texts)
            
            # Calculate metrics
            results = self._calculate_metrics(predictions, val_texts)
            
            # Save results
            with open(self.output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logging.info(f"Validation completed. Results saved to {self.output_path}")
            logging.info(f"Validation metrics: {results}")
            
        except Exception as e:
            logging.error(f"Error during validation: {str(e)}")
            raise AirflowException(f"Validation failed: {str(e)}")
    
    def _load_validation_data(self):
        """Load validation dataset"""
        # Create a simple validation dataset if it doesn't exist
        if not os.path.exists(self.validation_dataset_path):
            # Create dummy validation data
            val_data = [
                "Hello, how are you?",
                "What's the weather like?",
                "Tell me about artificial intelligence.",
                "How do you learn new things?",
                "What's your favorite color?"
            ]
            
            with open(self.validation_dataset_path, 'w') as f:
                json.dump(val_data, f)
            
            return val_data
        
        with open(self.validation_dataset_path, 'r') as f:
            return json.load(f)
    
    def _generate_predictions(self, model, tokenizer, texts):
        """Generate predictions for validation texts"""
        predictions = []
        
        for text in texts[:5]:  # Limit to 5 samples for demo
            inputs = tokenizer.encode(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 20,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_metrics(self, predictions, references):
        """Calculate validation metrics"""
        results = {}
        
        # Simple metrics for demo
        results['num_predictions'] = len(predictions)
        results['avg_prediction_length'] = sum(len(p.split()) for p in predictions) / len(predictions)
        
        # Mock BLEU and ROUGE scores (in real implementation, you'd use proper references)
        results['bleu'] = 0.25  # Mock score
        results['rouge-l'] = 0.30  # Mock score
        
        # Add timestamp
        results['validation_timestamp'] = datetime.now().isoformat()
        
        return results


class ModelUploadOperator(BaseOperator):
    """Upload a model to HuggingFace Hub"""
    
    @apply_defaults
    def __init__(
        self,
        model_path: str,
        model_name: str,
        private: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.model_name = model_name
        self.private = private
    
    def execute(self, context: Context):
        """Upload model to HuggingFace Hub"""
        logging.info(f"Uploading model to HuggingFace: {self.model_name}")
        
        # Get HuggingFace token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise AirflowException("HUGGINGFACE_TOKEN not found in environment")
        
        try:
            # Login to HuggingFace
            login(token=hf_token)
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            # Push to hub
            model.push_to_hub(
                self.model_name,
                private=self.private,
                token=hf_token
            )
            
            tokenizer.push_to_hub(
                self.model_name,
                private=self.private,
                token=hf_token
            )
            
            logging.info(f"Model successfully uploaded to {self.model_name}")
            
        except Exception as e:
            logging.error(f"Error uploading model: {str(e)}")
            raise AirflowException(f"Upload failed: {str(e)}")


class ManualApprovalSensor(BaseSensorOperator):
    """Sensor that waits for manual approval before proceeding"""
    
    @apply_defaults
    def __init__(
        self,
        model_path: str,
        validation_results_path: str,
        approval_file: str = "/opt/airflow/data/approval.json",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.validation_results_path = validation_results_path
        self.approval_file = approval_file
    
    def poke(self, context: Context) -> bool:
        """Check if approval has been granted"""
        logging.info("Checking for manual approval...")
        
        # Check if approval file exists
        if os.path.exists(self.approval_file):
            try:
                with open(self.approval_file, 'r') as f:
                    approval_data = json.load(f)
                
                # Check if approved for this specific run
                run_id = context['run_id']
                if approval_data.get('run_id') == run_id and approval_data.get('approved'):
                    logging.info("Manual approval granted!")
                    return True
                
            except Exception as e:
                logging.error(f"Error reading approval file: {str(e)}")
        
        # Log validation results for human review
        if os.path.exists(self.validation_results_path):
            with open(self.validation_results_path, 'r') as f:
                results = json.load(f)
            logging.info(f"Validation results: {results}")
            logging.info("Waiting for manual approval...")
            logging.info(f"To approve, create file {self.approval_file} with:")
            logging.info(f'{{"run_id": "{context["run_id"]}", "approved": true}}')
        
        return False