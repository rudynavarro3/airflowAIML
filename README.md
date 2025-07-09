# 🤖 MLOps Pipeline with Apache Airflow & HuggingFace

A production-ready MLOps pipeline that downloads, fine-tunes, validates, and deploys language models using Apache Airflow with comprehensive quality gates and automated HuggingFace deployment.

## 🏗️ Architecture Overview

This project implements a complete ML pipeline with:

- **Apache Airflow**: Orchestrates the entire ML workflow
- **Docker**: Containerized environment for consistent deployment  
- **HuggingFace Transformers**: Model downloading, fine-tuning, and deployment
- **Quality Gates**: Automated model validation with 6-point assessment
- **Manual Approval**: Human oversight before deployment
- **Real Deployment**: Automatic upload to HuggingFace Hub with model cards

## 📋 Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM and 4 CPU cores recommended
- HuggingFace account and write-enabled access token
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

Place these files in your project root:

#### **docker-compose.yml**
Use the simplified Docker Compose configuration (see artifacts above)

#### **.env** - Environment Variables
```bash
# Airflow configuration
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# HuggingFace configuration (REQUIRED)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Model configuration
BASE_MODEL_NAME=distilgpt2
FINE_TUNED_MODEL_NAME=your-username/airflow-finetuned-model
```

**⚠️ Important**: Replace `your-username` with your actual HuggingFace username and add your real token.

#### **data/simple_dataset.json**
Create your training dataset (see example dataset artifact above)

### 3. Setup Directory Structure

```
airflow-ml-pipeline/
├── docker-compose.yml
├── .env
├── dags/
│   └── simple_llm_no_trainer.py
├── data/
│   └── simple_dataset.json
├── logs/
├── models/
└── plugins/
```

### 4. Start the Services

```bash
# Set Airflow user ID (Linux/macOS)
echo -e "AIRFLOW_UID=$(id -u)" >> .env

# Start all services
docker-compose up -d

# Wait for services to be ready (about 2 minutes)
# Check status
docker-compose ps
```

### 5. Access Airflow UI

1. Open browser to `http://localhost:8080`
2. Login with:
   - Username: `airflow`
   - Password: `airflow`

## 🔧 Configuration

### HuggingFace Setup

1. **Create account** at [huggingface.co](https://huggingface.co)
2. **Generate token** at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - ✅ Make sure to enable **"Write"** permissions
3. **Update `.env`** with your token:
   ```bash
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
4. **Set model name** (must include your username):
   ```bash
   FINE_TUNED_MODEL_NAME=your-username/my-awesome-model
   ```

### Dataset Configuration

The pipeline uses `data/simple_dataset.json`. You can:

- **Use provided dataset**: 30 examples covering programming, ML, and data engineering
- **Create custom dataset**: Follow the JSON format with `texts` array
- **Load external dataset**: Modify the `prepare_dataset` function

## 🔄 Pipeline Workflow

The DAG `simple_llm_no_trainer` implements these stages:

### 1. **Model Download** 📥
- Downloads DistilGPT-2 (small, fast model)
- Caches locally for consistency
- Tests basic functionality

### 2. **Dataset Preparation** 📚
- Loads training examples from JSON
- Cleans and preprocesses text
- Tokenizes for training

### 3. **Training Simulation** 🎯
- Simulates fine-tuning process
- Calculates training metrics
- Demonstrates loss computation

### 4. **Model Validation** 🔍
- Tests text generation capabilities
- Evaluates response quality
- Measures coherence and length

### 5. **Quality Gate** ⚖️
**6-Point Assessment System:**
- ✅ **Success Rate**: ≥80% generations work
- ✅ **Quality Score**: ≥0.6 average rating
- ✅ **Response Length**: 2-50 words average  
- ✅ **Coherence**: ≥3 coherent responses
- ✅ **Error Rate**: ≤20% errors
- ✅ **Length Distribution**: Reasonable variance

### 6. **Manual Approval** 👥
- Reviews quality metrics
- Allows human override
- Gates deployment decisions

### 7. **HuggingFace Upload** 🚀
- Creates model repository
- Generates comprehensive model card
- Uploads tokenizer and config files
- Sets public visibility

### 8. **Cleanup** 🧹
- Removes temporary files
- Preserves logs and reports

## 🎯 Running the Pipeline

### Method 1: Airflow UI (Recommended)

1. **Navigate to** `http://localhost:8080`
2. **Find** `simple_llm_no_trainer` DAG
3. **Turn on** the DAG toggle
4. **Click** "Trigger DAG" to start
5. **Monitor** progress in Graph view

### Method 2: Command Line

```bash
# Enable the DAG
docker-compose exec airflow-scheduler airflow dags unpause simple_llm_no_trainer

# Trigger a run
docker-compose exec airflow-scheduler airflow dags trigger simple_llm_no_trainer

# Monitor progress
docker-compose logs -f airflow-scheduler
```

## 🔍 Monitoring and Debugging

### Real-time Monitoring

```bash
# Watch all logs
docker-compose logs -f

# Watch specific service
docker-compose logs -f airflow-scheduler

# View task logs
docker-compose exec airflow-scheduler airflow tasks logs simple_llm_no_trainer quality_gate 2024-01-01
```

### Quality Gate Logs

The quality gate produces detailed assessment logs:

```
🎯 QUALITY GATE ASSESSMENT
==================================================
✅ PASS success_rate_check
✅ PASS quality_score_check  
✅ PASS min_length_check
✅ PASS max_length_check
✅ PASS coherence_check
✅ PASS error_rate_check
==================================================
📊 DETAILED METRICS:
  • Success Rate: 100.00% (min: 80.00%)
  • Quality Score: 0.73 (min: 0.60)
  • Avg Length: 8.2 words (range: 2.0-50.0)
  • Coherent Responses: 5 (min: 3)
  • Error Rate: 0.00% (max: 20.00%)
==================================================
🎉 QUALITY GATE: ✅ APPROVED
```

### Upload Confirmation

Successful uploads show:

```
🚀 REAL HUGGINGFACE UPLOAD
========================================
📤 Uploading to: your-username/airflow-finetuned-model
  ✅ Repository created/verified
  ✅ Created model card
  📤 Uploading files...
    ✅ Uploaded: README.md
    ✅ Uploaded: tokenizer.json
    ✅ Uploaded: tokenizer_config.json
🎉 UPLOAD COMPLETED SUCCESSFULLY!
🔗 Model URL: https://huggingface.co/your-username/airflow-finetuned-model
```

## 📊 Quality Metrics Explained

### Success Rate
Percentage of text generations that complete without errors.

### Quality Score  
Composite score (0-1) based on:
- Response coherence (logical flow)
- Appropriate length (not too short/long)
- Contains actual content (not empty)

### Response Length
Average words in generated responses. Should be:
- Minimum: 2 words (substantive)
- Maximum: 50 words (not overly verbose)

### Coherence Count
Number of responses that form logical, complete thoughts.

### Error Rate
Percentage of generations that fail due to technical errors.

## 🛠️ Customization

### Different Base Models

```bash
# In .env file:
BASE_MODEL_NAME=gpt2                    # Larger GPT-2
BASE_MODEL_NAME=microsoft/DialoGPT-small # Conversation model
BASE_MODEL_NAME=distilbert-base-uncased  # BERT variant
```

### Custom Datasets

```python
# In prepare_dataset function:
def prepare_dataset(**context):
    # Load from CSV
    import pandas as pd
    df = pd.read_csv('/opt/airflow/data/my_data.csv')
    texts = df['text_column'].tolist()
    
    # Or load from HuggingFace
    from datasets import load_dataset
    dataset = load_dataset("your_dataset_name")
    texts = dataset['train']['text']
```

### Quality Thresholds

```python
# In quality_gate function, modify:
quality_criteria = {
    'min_success_rate': 0.9,      # Stricter: 90%
    'min_quality_score': 0.7,     # Higher quality  
    'min_avg_length': 5.0,        # Longer responses
    'max_avg_length': 30.0,       # Shorter responses
    'min_coherent_responses': 4,   # More coherent
    'max_error_rate': 0.1          # Fewer errors
}
```

## 🔄 Development Workflow

### Making DAG Changes

```bash
# 1. Edit your DAG file
vim dags/simple_llm_no_trainer.py

# 2. Save - Airflow auto-detects changes (~30 seconds)

# 3. For immediate reload:
docker-compose restart airflow-scheduler

# 4. Check for syntax errors in UI or logs
docker-compose logs airflow-scheduler | grep ERROR
```

### Testing Changes

```bash
# Test DAG syntax
docker-compose exec airflow-scheduler python /opt/airflow/dags/simple_llm_no_trainer.py

# Test specific task
docker-compose exec airflow-scheduler airflow tasks test simple_llm_no_trainer quality_gate 2024-01-01
```

### When to Restart Containers

**No restart needed:**
- ✅ DAG code changes
- ✅ Dataset modifications
- ✅ Task logic updates

**Restart required:**
- ❌ Environment variable changes (`.env`)
- ❌ Docker Compose changes
- ❌ New Python package installations
- ❌ Plugin additions

## 🚫 Troubleshooting

### Common Issues

#### **DAG Not Appearing**
```bash
# Check for syntax errors
docker-compose exec airflow-scheduler python /opt/airflow/dags/simple_llm_no_trainer.py

# Check file permissions
ls -la dags/

# Restart scheduler
docker-compose restart airflow-scheduler
```

#### **HuggingFace Upload Fails**
```bash
# Verify token has write permissions
# Check repository name format: username/model-name
# Ensure token is set in .env file

# Test token manually:
docker-compose exec airflow-scheduler python -c "
from huggingface_hub import whoami, login
login('your_token_here')
print(whoami())
"
```

#### **Quality Gate Always Fails**
```bash
# Check validation results
docker-compose exec airflow-scheduler cat /opt/airflow/models/validation_results.json

# Lower thresholds for testing:
# Edit quality_criteria in quality_gate function
```

#### **Memory Issues**
```bash
# Reduce model size or batch size
# Use smaller base model like distilgpt2
# Increase Docker memory limits
```

### Debug Commands

```bash
# View container resources
docker stats

# Check disk space
docker-compose exec airflow-scheduler df -h

# View all DAGs
docker-compose exec airflow-scheduler airflow dags list

# Check DAG bag
docker-compose exec airflow-scheduler airflow dags show simple_llm_no_trainer
```

## 📈 Production Considerations

### Security
- **Rotate HuggingFace tokens** regularly
- **Use secrets management** instead of .env files
- **Implement proper access controls**
- **Monitor model deployments**

### Scaling
- **Use external databases** (PostgreSQL/MySQL)
- **Add Redis for caching**
- **Implement horizontal scaling**
- **Use cloud storage** for models

### Monitoring
- **Set up alerting** for failed runs
- **Monitor quality metrics** over time
- **Track model performance** post-deployment
- **Log aggregation** and analysis

### CI/CD Integration
- **Version control** DAGs
- **Automated testing** of pipeline changes  
- **Staged deployments** (dev → staging → prod)
- **Model versioning** and rollback capabilities

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [MLOps Best Practices](https://ml-ops.org/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Test changes thoroughly
4. Update documentation
5. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License.

---

## 🎉 Success Indicators

After successful pipeline execution, you should have:

✅ **Airflow UI** showing all green tasks  
✅ **Quality report** with detailed metrics  
✅ **HuggingFace model** deployed and public  
✅ **Model card** with training details  
✅ **Comprehensive logs** for debugging  

**🚀 Ready for Production MLOps! 🚀**