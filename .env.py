# Airflow configuration
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# HuggingFace configuration
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model configuration
BASE_MODEL_NAME=microsoft/DialoGPT-small
FINE_TUNED_MODEL_NAME=your-username/dialogpt-small-finetuned
DATASET_NAME=daily_dialog

# Training parameters
BATCH_SIZE=8
LEARNING_RATE=5e-5
NUM_EPOCHS=3
MAX_LENGTH=512