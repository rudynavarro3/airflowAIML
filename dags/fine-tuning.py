from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
import os
import logging
from pathlib import Path

# Import custom operators
from plugins.ml_operators import (
    ModelDownloadOperator,
    ModelFineTuneOperator,
    ModelValidationOperator,
    ModelUploadOperator,
    ManualApprovalSensor
)

# Default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

# DAG configuration
dag = DAG(
    'ml_model_finetune_pipeline',
    default_args=default_args,
    description='Fine-tune HuggingFace model with manual approval gate',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'huggingface', 'fine-tuning'],
)

# Configuration
BASE_MODEL = Variable.get("base_model_name", "microsoft/DialoGPT-small")
DATASET_NAME = Variable.get("dataset_name", "daily_dialog")
FINE_TUNED_MODEL = Variable.get("fine_tuned_model_name", "your-username/dialogpt-small-finetuned")

# Task 1: Download base model from HuggingFace
download_model = ModelDownloadOperator(
    task_id='download_base_model',
    model_name=BASE_MODEL,
    local_path='/opt/airflow/models/base_model',
    dag=dag
)

# Task 2: Prepare dataset
def prepare_dataset(**context):
    """Prepare the dataset for fine-tuning"""
    from datasets import load_dataset
    import pandas as pd
    
    logging.info(f"Loading dataset: {DATASET_NAME}")
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split='train[:1000]')  # Small subset for demo
    
    # Save to local path
    dataset_path = '/opt/airflow/data/training_data.json'
    dataset.to_json(dataset_path)
    
    logging.info(f"Dataset saved to {dataset_path}")
    logging.info(f"Dataset size: {len(dataset)}")
    
    return dataset_path

prepare_data = PythonOperator(
    task_id='prepare_dataset',
    python_callable=prepare_dataset,
    dag=dag
)

# Task 3: Fine-tune the model
finetune_model = ModelFineTuneOperator(
    task_id='finetune_model',
    base_model_path='/opt/airflow/models/base_model',
    dataset_path='/opt/airflow/data/training_data.json',
    output_path='/opt/airflow/models/finetuned_model',
    training_args={
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'num_train_epochs': 2,
        'learning_rate': 5e-5,
        'warmup_steps': 100,
        'logging_steps': 10,
        'save_strategy': 'epoch',
        'evaluation_strategy': 'epoch',
        'load_best_model_at_end': True,
    },
    dag=dag
)

# Task 4: Validate the fine-tuned model
validate_model = ModelValidationOperator(
    task_id='validate_finetuned_model',
    model_path='/opt/airflow/models/finetuned_model',
    validation_dataset_path='/opt/airflow/data/validation_data.json',
    metrics=['bleu', 'rouge'],
    output_path='/opt/airflow/models/validation_results.json',
    dag=dag
)

# Task 5: Manual approval gate
def check_validation_results(**context):
    """Check if validation results meet criteria"""
    import json
    
    results_path = '/opt/airflow/models/validation_results.json'
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Define thresholds
    min_bleu = 0.15
    min_rouge = 0.25
    
    bleu_score = results.get('bleu', 0)
    rouge_score = results.get('rouge-l', 0)
    
    logging.info(f"Validation Results - BLEU: {bleu_score}, ROUGE-L: {rouge_score}")
    
    # Check if model meets minimum criteria
    if bleu_score >= min_bleu and rouge_score >= min_rouge:
        logging.info("Model meets minimum quality criteria")
        return True
    else:
        logging.warning("Model does not meet minimum quality criteria")
        return False

check_quality = PythonOperator(
    task_id='check_model_quality',
    python_callable=check_validation_results,
    dag=dag
)

# Task 6: Manual approval sensor
manual_approval = ManualApprovalSensor(
    task_id='manual_approval_gate',
    model_path='/opt/airflow/models/finetuned_model',
    validation_results_path='/opt/airflow/models/validation_results.json',
    timeout=timedelta(days=7),  # Wait up to 7 days for approval
    poke_interval=300,  # Check every 5 minutes
    dag=dag
)

# Task 7: Upload to HuggingFace (only if approved)
upload_model = ModelUploadOperator(
    task_id='upload_to_huggingface',
    model_path='/opt/airflow/models/finetuned_model',
    model_name=FINE_TUNED_MODEL,
    private=False,  # Make public
    dag=dag
)

# Task 8: Cleanup temporary files
def cleanup_files(**context):
    """Clean up temporary files"""
    import shutil
    
    paths_to_clean = [
        '/opt/airflow/models/base_model',
        '/opt/airflow/models/finetuned_model',
        '/opt/airflow/data/training_data.json',
        '/opt/airflow/data/validation_data.json'
    ]
    
    for path in paths_to_clean:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            logging.info(f"Cleaned up: {path}")

cleanup = PythonOperator(
    task_id='cleanup_temporary_files',
    python_callable=cleanup_files,
    trigger_rule=TriggerRule.ALL_DONE,  # Run regardless of upstream success/failure
    dag=dag
)

# Task 9: Notification task
def send_notification(**context):
    """Send notification about pipeline completion"""
    logging.info("ML Pipeline completed successfully!")
    logging.info(f"Model uploaded to: {FINE_TUNED_MODEL}")
    
    # Here you could integrate with Slack, email, etc.
    # For now, just log the completion

notification = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_notification,
    dag=dag
)

# Define task dependencies
download_model >> prepare_data >> finetune_model >> validate_model >> check_quality

# Branch based on quality check
check_quality >> manual_approval >> upload_model >> notification
upload_model >> cleanup
manual_approval >> cleanup  # Also cleanup if approval times out

# Set up the pipeline flow
# download_model → prepare_data → finetune_model → validate_model → check_quality → manual_approval → upload_model → notification
#                                                                                                  ↓
#                                                                                               cleanup