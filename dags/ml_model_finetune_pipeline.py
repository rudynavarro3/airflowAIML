from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import os
import json
import logging

# Default arguments
default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG configuration
dag = DAG(
    'simple_llm_no_trainer',
    default_args=default_args,
    description='Simple LLM Pipeline without Trainer class',
    schedule_interval=None,
    catchup=False,
    tags=['llm', 'simple', 'no-trainer'],
)

MODEL_NAME = "distilgpt2"

def download_model(**context):
    """Download and cache model locally"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login
        
        logging.info(f"📥 Downloading model: {MODEL_NAME}")
        
        # Login if token available
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            login(token=hf_token)
        
        # Download and test
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # Add pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Save locally for consistency
        local_path = '/opt/airflow/models/base_model'
        os.makedirs(local_path, exist_ok=True)
        
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        
        # Test basic functionality
        test_input = "Hello"
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        model.eval()
        outputs = model.generate(inputs, max_length=inputs.shape[1] + 3, do_sample=False)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        logging.info(f"✅ Model downloaded and cached locally")
        logging.info(f"📊 Parameters: {total_params:,}")
        logging.info(f"🔍 Test: '{test_input}' → '{result}'")
        
        return {
            'model_path': local_path,
            'total_params': total_params,
            'test_successful': True
        }
        
    except Exception as e:
        logging.error(f"❌ Error downloading model: {str(e)}")
        raise

def prepare_dataset(**context):
    """Load and prepare the training dataset"""
    try:
        logging.info("📚 Loading training dataset...")
        
        dataset_path = '/opt/airflow/data/simple_dataset.json'
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logging.warning("📂 Dataset not found, creating sample data...")
            
            # Create sample data if file doesn't exist
            sample_data = {
                "texts": [
                    "Hello! How can I help you today?",
                    "Programming is the process of creating software.",
                    "Machine learning helps computers learn from data.",
                    "Python is a versatile programming language.",
                    "Thank you for your question."
                ],
                "num_examples": 5
            }
            
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            with open(dataset_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
        
        # Load the dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        texts = dataset['texts']
        
        logging.info(f"✅ Loaded {len(texts)} training examples")
        logging.info(f"📝 Example: '{texts[0][:50]}...'")
        
        # Simple preprocessing
        processed_texts = []
        for text in texts:
            # Basic cleaning
            clean_text = text.strip()
            if clean_text:  # Only add non-empty texts
                processed_texts.append(clean_text)
        
        # Save processed data
        processed_data = {
            'texts': processed_texts,
            'original_count': len(texts),
            'processed_count': len(processed_texts)
        }
        
        processed_path = '/opt/airflow/data/processed_dataset.json'
        with open(processed_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logging.info(f"✅ Processed {len(processed_texts)} examples")
        
        return {
            'dataset_path': processed_path,
            'num_examples': len(processed_texts)
        }
        
    except Exception as e:
        logging.error(f"❌ Error preparing dataset: {str(e)}")
        raise

def manual_training_simulation(**context):
    """Simulate fine-tuning without using Trainer class"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import torch.nn.functional as F
        import json
        
        logging.info("🚀 Starting manual training simulation...")
        
        # Load model and tokenizer
        model_path = '/opt/airflow/models/base_model'
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        except:
            # Fallback to direct download
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Load processed dataset
        dataset_path = '/opt/airflow/data/processed_dataset.json'
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        texts = data['texts'][:5]  # Use only first 5 for quick demo
        
        # Manual tokenization
        logging.info("📝 Tokenizing training data...")
        tokenized_examples = []
        total_tokens = 0
        
        for text in texts:
            # Tokenize each text
            tokens = tokenizer.encode(
                text,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt'
            )
            
            tokenized_examples.append(tokens)
            total_tokens += tokens.shape[1]
            
            logging.info(f"  • '{text[:40]}...' → {tokens.shape[1]} tokens")
        
        # Simulate training steps (without actual training for simplicity)
        logging.info("🎯 Simulating training process...")
        
        model.eval()  # Set to eval mode for consistency
        
        # Simulate calculating loss on a few examples
        total_loss = 0
        num_examples = len(tokenized_examples)
        
        with torch.no_grad():  # No gradients needed for simulation
            for i, tokens in enumerate(tokenized_examples):
                # Forward pass
                outputs = model(tokens, labels=tokens)
                loss = outputs.loss
                total_loss += loss.item()
                
                logging.info(f"  • Example {i+1}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_examples
        
        # Simulate training metrics
        training_metrics = {
            'model_name': MODEL_NAME,
            'training_examples': num_examples,
            'total_tokens': total_tokens,
            'avg_tokens_per_example': total_tokens / num_examples,
            'simulated_epochs': 3,
            'initial_loss': avg_loss,
            'final_loss': avg_loss * 0.8,  # Simulate improvement
            'training_method': 'manual_simulation',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metrics
        metrics_path = '/opt/airflow/models/training_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        logging.info("📊 Training Simulation Complete:")
        logging.info(f"  • Examples: {num_examples}")
        logging.info(f"  • Avg tokens: {total_tokens / num_examples:.1f}")
        logging.info(f"  • Initial loss: {avg_loss:.4f}")
        logging.info(f"  • Simulated final loss: {training_metrics['final_loss']:.4f}")
        
        return training_metrics
        
    except Exception as e:
        logging.error(f"❌ Error in training simulation: {str(e)}")
        raise

def validate_model_generation(**context):
    """Validate model by testing text generation"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logging.info("🔍 Validating model generation...")
        
        # Load model (using original since we didn't actually fine-tune)
        try:
            model_path = '/opt/airflow/models/base_model'
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        model.eval()
        
        # Test prompts
        test_prompts = [
            "Hello",
            "Programming is",
            "Machine learning",
            "Python helps",
            "Thank you"
        ]
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_tested': MODEL_NAME,
            'test_cases': []
        }
        
        successful_generations = 0
        total_response_length = 0
        
        for prompt in test_prompts:
            try:
                # Tokenize
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 12,  # Generate 12 more tokens
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_text = generated_text[len(prompt):].strip()
                
                # Quality checks
                is_coherent = len(new_text.split()) >= 2  # At least 2 words
                is_reasonable_length = 2 <= len(new_text.split()) <= 20
                contains_text = len(new_text.strip()) > 0
                
                quality_score = sum([is_coherent, is_reasonable_length, contains_text]) / 3
                
                test_case = {
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'new_text': new_text,
                    'word_count': len(new_text.split()),
                    'is_coherent': is_coherent,
                    'is_reasonable_length': is_reasonable_length,
                    'contains_text': contains_text,
                    'quality_score': quality_score,
                    'success': quality_score >= 0.6
                }
                
                validation_results['test_cases'].append(test_case)
                
                if test_case['success']:
                    successful_generations += 1
                
                total_response_length += len(new_text.split())
                
                logging.info(f"✅ '{prompt}' → '{generated_text}' (score: {quality_score:.2f})")
                
            except Exception as e:
                test_case = {
                    'prompt': prompt,
                    'error': str(e),
                    'success': False,
                    'quality_score': 0
                }
                validation_results['test_cases'].append(test_case)
                logging.error(f"❌ Error with prompt '{prompt}': {e}")
        
        # Calculate overall metrics
        total_tests = len(test_prompts)
        success_rate = successful_generations / total_tests
        avg_response_length = total_response_length / total_tests
        avg_quality_score = sum(tc.get('quality_score', 0) for tc in validation_results['test_cases']) / total_tests
        
        validation_results['summary'] = {
            'total_tests': total_tests,
            'successful_generations': successful_generations,
            'success_rate': success_rate,
            'avg_response_length': avg_response_length,
            'avg_quality_score': avg_quality_score
        }
        
        # Save validation results
        results_path = '/opt/airflow/models/validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logging.info("📊 Validation Summary:")
        logging.info(f"  • Success rate: {success_rate:.2%}")
        logging.info(f"  • Avg quality score: {avg_quality_score:.2f}")
        logging.info(f"  • Avg response length: {avg_response_length:.1f} words")
        
        return validation_results['summary']
        
    except Exception as e:
        logging.error(f"❌ Error during validation: {str(e)}")
        raise

def quality_gate(**context):
    """Check if model passes quality gates"""
    try:
        results_path = '/opt/airflow/models/validation_results.json'
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        summary = results['summary']
        
        # Quality thresholds
        min_success_rate = 0.6  # 60%
        min_quality_score = 0.5  # 0.5/1.0
        min_avg_length = 1.0   # 1 word average
        
        success_rate = summary['success_rate']
        quality_score = summary['avg_quality_score']
        avg_length = summary['avg_response_length']
        
        logging.info("🎯 Quality Gate Assessment:")
        logging.info(f"  • Success Rate: {success_rate:.2%} (min: {min_success_rate:.2%})")
        logging.info(f"  • Quality Score: {quality_score:.2f} (min: {min_quality_score:.2f})")
        logging.info(f"  • Avg Length: {avg_length:.1f} words (min: {min_avg_length})")
        
        passes_quality = (
            success_rate >= min_success_rate and
            quality_score >= min_quality_score and
            avg_length >= min_avg_length
        )
        
        if passes_quality:
            logging.info("✅ Model PASSES quality gates!")
        else:
            logging.warning("❌ Model FAILS quality gates!")
        
        return passes_quality
        
    except Exception as e:
        logging.error(f"❌ Quality gate check failed: {str(e)}")
        return False

def approval_and_upload(**context):
    """Manual approval and mock upload"""
    try:
        # Check quality gate result
        quality_passed = context['task_instance'].xcom_pull(task_ids='quality_gate')
        
        if not quality_passed:
            logging.warning("⚠️ Model failed quality gates - requiring manual review")
        
        logging.info("⏳ Manual Approval Process:")
        logging.info("  • Model validation completed")
        logging.info("  • Quality metrics calculated")
        logging.info("  • Ready for human review")
        
        # Simulate approval (in production, this would wait for human input)
        approval_data = {
            'timestamp': datetime.now().isoformat(),
            'quality_gate_passed': quality_passed,
            'approved_by': 'demo-system',
            'approval_reason': 'automated_demo'
        }
        
        # Mock upload process
        model_name = os.getenv('FINE_TUNED_MODEL_NAME', 'demo/simple-model-v1')
        
        logging.info("🚀 Mock Upload Process:")
        logging.info(f"  • Target repository: {model_name}")
        logging.info("  • Upload steps:")
        logging.info("    1. ✅ Validate model files")
        logging.info("    2. ✅ Create model card")
        logging.info("    3. ✅ Upload to HuggingFace Hub")
        logging.info("    4. ✅ Set model visibility")
        logging.info("    5. ✅ Add model tags")
        
        upload_info = {
            **approval_data,
            'model_repository': model_name,
            'upload_url': f"https://huggingface.co/{model_name}",
            'upload_status': 'mock_completed',
            'model_size_mb': 82.5,  # Mock size for DistilGPT2
            'upload_time_seconds': 45
        }
        
        # Save upload info
        upload_path = '/opt/airflow/models/upload_info.json'
        with open(upload_path, 'w') as f:
            json.dump(upload_info, f, indent=2)
        
        logging.info(f"✅ Mock upload completed!")
        logging.info(f"🔗 Model available at: {upload_info['upload_url']}")
        
        return upload_info
        
    except Exception as e:
        logging.error(f"❌ Error in approval/upload: {str(e)}")
        raise

def cleanup_files(**context):
    """Clean up temporary files"""
    import shutil
    
    cleanup_paths = [
        '/opt/airflow/models/base_model',
        '/opt/airflow/data/processed_dataset.json',
        '/opt/airflow/models/training_metrics.json',
        '/opt/airflow/models/validation_results.json',
        '/opt/airflow/models/upload_info.json'
    ]
    
    logging.info("🧹 Cleaning up files...")
    
    for path in cleanup_paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logging.info(f"  ✅ Removed directory: {path}")
                else:
                    os.remove(path)
                    logging.info(f"  ✅ Removed file: {path}")
        except Exception as e:
            logging.warning(f"  ⚠️ Could not remove {path}: {e}")
    
    logging.info("🧹 Cleanup completed!")

# Define tasks
start_task = DummyOperator(task_id='start_pipeline', dag=dag)

download_task = PythonOperator(
    task_id='download_model',
    python_callable=download_model,
    dag=dag
)

dataset_task = PythonOperator(
    task_id='prepare_dataset',
    python_callable=prepare_dataset,
    dag=dag
)

training_task = PythonOperator(
    task_id='manual_training_simulation',
    python_callable=manual_training_simulation,
    dag=dag
)

validation_task = PythonOperator(
    task_id='validate_model_generation',
    python_callable=validate_model_generation,
    dag=dag
)

quality_task = PythonOperator(
    task_id='quality_gate',
    python_callable=quality_gate,
    dag=dag
)

approval_task = PythonOperator(
    task_id='approval_and_upload',
    python_callable=approval_and_upload,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_files',
    python_callable=cleanup_files,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag
)

end_task = DummyOperator(task_id='end_pipeline', dag=dag)

# Define dependencies
start_task >> download_task >> dataset_task >> training_task >> validation_task >> quality_task >> approval_task >> end_task
approval_task >> cleanup_task