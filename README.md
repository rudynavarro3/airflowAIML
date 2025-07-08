## 🔄 LLM Fine-tuning Pipeline Workflow

The DAG `llm_finetune_pipeline` implements modern LLM fine-tuning techniques:

1. **Download Base LLM**: Downloads pre-trained language model from HuggingFace
2. **Prepare Instruction Dataset**: Creates instruction-following dataset format
3. **Setup LoRA Configuration**: Configures Low-Rank Adaptation parameters
4. **LoRA# ML Model Fine-tuning Pipeline with Apache Airflow

A complete MLOps pipeline that downloads a model from HuggingFace, fine-tunes it, validates the results, and uploads the improved model back to HuggingFace with manual approval gates.

## 🏗️ Architecture Overview

This project demonstrates a production-ready ML pipeline with the following components:

- **Apache Airflow**: Orchestrates the entire ML workflow
- **Docker**: Containerized environment for consistent deployment
- **HuggingFace Transformers**: Model downloading, fine-tuning, and uploading
- **PostgreSQL**: Airflow metadata database
- **Manual Approval Gates**: Human oversight before model deployment

## 📋 Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM and 4 CPU cores recommended
- HuggingFace account and access token
- Basic understanding of Python and ML concepts

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir airflow-ml-pipeline
cd airflow-ml-pipeline

# Create required directories
mkdir -p dags logs plugins models data
```

### 2. Create Configuration Files

Create the following files in your project root:

**docker-compose.yml** - Main Airflow setup (see artifact above)

**requirements.txt** - Python dependencies (see artifact above)

**.env** - Environment variables:
```bash
# Copy the .env template and update with your values
cp .env.example .env

# Edit .env with your HuggingFace token
HUGGINGFACE_TOKEN=your_actual_token_here
BASE_MODEL_NAME=microsoft/DialoGPT-small
FINE_TUNED_MODEL_NAME=your-username/dialogpt-small-finetuned
```

### 3. Setup Directory Structure

```
airflow-ml-pipeline/
├── docker-compose.yml
├── requirements.txt
├── .env
├── dags/
│   └── ml_model_finetune_pipeline.py
├── plugins/
│   └── ml_operators.py
├── logs/
├── models/
└── data/
```

### 4. Start the Services

```bash
# Set Airflow user ID (Linux/macOS)
echo -e "AIRFLOW_UID=$(id -u)" >> .env

# Initialize Airflow database and create admin user
docker-compose up airflow-init

# Start all Airflow services
docker-compose up -d
```

### 5. Access Airflow UI

1. Open browser to `http://localhost:8080`
2. Login with:
   - Username: `airflow`
   - Password: `airflow`

## 🔧 Configuration

### HuggingFace Setup

1. Create account at [huggingface.co](https://huggingface.co)
2. Generate access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Update `.env` file with your token:
   ```bash
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### Model Configuration

Update these variables in `.env` or Airflow Variables:

```bash
# Base model to fine-tune
BASE_MODEL_NAME=microsoft/DialoGPT-small

# Your fine-tuned model name (must include your username)
FINE_TUNED_MODEL_NAME=your-username/dialogpt-small-finetuned

# Dataset for training
DATASET_NAME=daily_dialog

# Training parameters
BATCH_SIZE=8
LEARNING_RATE=5e-5
NUM_EPOCHS=3
MAX_LENGTH=512
```

## 🔄 Pipeline Workflow

The DAG `ml_model_finetune_pipeline` includes these tasks:

1. **Download Base Model**: Downloads specified model from HuggingFace
2. **Prepare Dataset**: Loads and preprocesses training data
3. **Fine-tune Model**: Trains the model with your data
4. **Validate Model**: Runs validation metrics (BLEU, ROUGE)
5. **Quality Check**: Ensures model meets minimum criteria
6. **Manual Approval**: Human gate before deployment
7. **Upload Model**: Pushes approved model to HuggingFace
8. **Cleanup**: Removes temporary files
9. **Notification**: Sends completion alerts

## 🎯 Running the Pipeline

### Method 1: Airflow UI

1. Navigate to `http://localhost:8080`
2. Find `ml_model_finetune_pipeline` DAG
3. Toggle the DAG to "On"
4. Click "Trigger DAG" to start manually

### Method 2: CLI

```bash
# Enable the DAG
docker-compose exec airflow-worker airflow dags unpause ml_model_finetune_pipeline

# Trigger a run
docker-compose exec airflow-worker airflow dags trigger ml_model_finetune_pipeline

# Check DAG status
docker-compose exec airflow-worker airflow dags state ml_model_finetune_pipeline 2024-01-01
```

## 🔍 Monitoring and Debugging

### Check Logs

```bash
# View all logs
docker-compose logs airflow-scheduler

# View specific task logs
docker-compose exec airflow-worker airflow tasks logs ml_model_finetune_pipeline download_base_model 2024-01-01

# Follow logs in real-time
docker-compose logs -f airflow-worker
```

### Access Container

```bash
# Access worker container
docker-compose exec airflow-worker bash

# Check file system
ls -la /opt/airflow/models/
ls -la /opt/airflow/data/
```

### Common Issues

**GPU Support**: To enable GPU training, modify `docker-compose.yml`:
```yaml
services:
  airflow-worker:
    # Add GPU support
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Memory Issues**: Reduce batch size in training arguments or increase Docker memory limits.

**HuggingFace Authentication**: Ensure your token has write permissions for model uploads.

## 🔐 Manual Approval Process

The pipeline includes a manual approval gate that:

1. **Waits for Approval**: Task pauses after validation
2. **Shows Results**: Validation metrics available in Airflow UI
3. **Approval Methods**:
   - Mark task as "Success" in Airflow UI
   - Use CLI: `airflow tasks mark-success`
   - Custom approval interface (extend `ManualApprovalSensor`)

### Approve via UI
1. Go to DAG run view
2. Click on `manual_approval_gate` task
3. Click "Mark Success" to approve
4. Pipeline continues to upload

### Approve via CLI
```bash
docker-compose exec airflow-worker airflow tasks mark-success \
  ml_model_finetune_pipeline manual_approval_gate 2024-01-01
```

## 📊 Customization

### Different Models
Change the base model in `.env`:
```bash
BASE_MODEL_NAME=gpt2
BASE_MODEL_NAME=microsoft/DialoGPT-medium
BASE_MODEL_NAME=facebook/blenderbot-400M-distill
```

### Custom Datasets
Modify `prepare_dataset` function in the DAG to use your data:
```python
def prepare_dataset(**context):
    # Load your custom dataset
    dataset = load_dataset("your_dataset_name")
    # or load from local files
    # dataset = Dataset.from_csv("your_data.csv")
```

### Training Parameters
Adjust training in `finetune_model` task:
```python
training_args={
    'per_device_train_batch_size': 8,  # Increase for more memory
    'num_train_epochs': 5,             # More epochs for better training
    'learning_rate': 3e-5,             # Lower LR for stability
    'warmup_steps': 500,               # More warmup steps
}
```

## 🛑 Stopping and Cleanup

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

### Clean Docker
```bash
# Remove all containers and images
docker system prune -a

# Remove specific containers
docker-compose rm
```

## 🔧 Troubleshooting

### Port Conflicts
If port 8080 is busy:
```yaml
# In docker-compose.yml, change webserver ports
ports:
  - "8081:8080"  # Use port 8081 instead
```

### Permission Issues
```bash
# Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER logs/
chmod -R 755 logs/
```

### Database Issues
```bash
# Reset Airflow database
docker-compose down -v
docker-compose up airflow-init
docker-compose up -d
```

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [MLOps Best Practices](https://ml-ops.org/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Happy ML Engineering! 🚀**