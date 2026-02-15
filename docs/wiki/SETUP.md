# ⚙️ Setup & Deployment Guide

Complete guide for setting up the RAG Second Brain development environment and deploying to production.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Anirach/rag-second-brain.git
cd rag-second-brain

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py
```

---

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- RAM: 4GB minimum, 8GB recommended
- Storage: 2GB free space
- CPU: 2+ cores recommended

**Recommended Requirements:**
- Python 3.9-3.11
- RAM: 16GB+ for large corpora
- Storage: 10GB+ for datasets and indexes
- GPU: CUDA-compatible for acceleration (optional)

### Platform Support

| Platform | Status | Notes |
|----------|---------|-------|
| Linux (Ubuntu 20.04+) | ✅ Fully Supported | Recommended for production |
| macOS (10.15+) | ✅ Fully Supported | Development and testing |
| Windows 10/11 | ✅ Supported | Some dependencies may need extra setup |
| Docker | ✅ Fully Supported | Cross-platform deployment |

---

## 🛠️ Development Environment Setup

### 1. Python Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv rag-second-brain-env
source rag-second-brain-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

#### Using conda
```bash
# Create conda environment
conda create -n rag-second-brain python=3.9
conda activate rag-second-brain

# Install PyTorch (if using GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Core Dependencies Installation

#### Sentence Transformers & FAISS
```bash
# CPU version (smaller, faster install)
pip install sentence-transformers faiss-cpu

# GPU version (better performance for large datasets)
pip install sentence-transformers faiss-gpu
```

#### Knowledge Graph Dependencies
```bash
# spaCy with language model
pip install spacy
python -m spacy download en_core_web_sm

# NetworkX for graph operations
pip install networkx

# Optional: graph visualization
pip install matplotlib plotly
```

#### Ontology Dependencies
```bash
# OWLready2 for ontology processing
pip install owlready2

# RDF processing
pip install rdflib

# Java (required for Pellet reasoner)
# Ubuntu/Debian: sudo apt-get install default-jdk
# macOS: brew install openjdk
# Windows: Download from Oracle or OpenJDK
```

### 3. Development Tools

```bash
# Code formatting and linting
pip install black isort flake8

# Testing
pip install pytest pytest-cov

# Jupyter notebooks (for experiments)
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=rag-second-brain
```

### 4. Environment Configuration

Create `.env` file in project root:
```bash
# Model Configuration
DENSE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
KG_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Parameters
COOCCUR_WINDOW_SIZE=5
COOCCUR_MIN_COUNT=2
DENSE_INDEX_TYPE=IndexFlatIP
KG_ENTITY_THRESHOLD=0.8

# Ontology Configuration
ONTOLOGY_PATH=data/ontologies/domain.owl
REASONING_DEPTH=3
USE_PELLET_REASONER=true

# Gating Network
GATING_HIDDEN_DIM=128
GATING_LEARNING_RATE=0.0001
GATING_DROPOUT=0.1

# Performance
BATCH_SIZE=32
NUM_WORKERS=4
CACHE_SIZE=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag_second_brain.log
```

### 5. Data Setup

#### Sample Data
```bash
# Create data directories
mkdir -p data/{sample,ontologies,models}

# Download sample corpus (if available)
# wget https://example.com/sample_corpus.zip
# unzip sample_corpus.zip -d data/sample/

# Create sample documents for testing
cat > data/sample/test_docs.txt << EOF
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand text.
Knowledge graphs represent information as interconnected entities.
EOF
```

#### Ontology Files
```bash
# Download or create domain ontology
# Example: Academic domain ontology
cat > data/ontologies/academic.owl << EOF
<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.com/academic#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    
    <owl:Ontology rdf:about="http://example.com/academic"/>
    
    <owl:Class rdf:about="#Document"/>
    <owl:Class rdf:about="#Person"/>
    <owl:Class rdf:about="#Organization"/>
    
    <owl:ObjectProperty rdf:about="#hasAuthor"/>
    <owl:ObjectProperty rdf:about="#affiliatedWith"/>
    
</rdf:RDF>
EOF
```

---

## 🏃‍♂️ Running the System

### 1. Menu-Driven Demo

```bash
# Activate environment
source venv/bin/activate

# Run main demo
python main.py
```

**Expected Output:**
```
==================================================
  RAG Second Brain - Proof of Concept
  Multi-Source Retrieval with Learned Gating
==================================================

--- Main Menu ---
1. Co-occurrence Module Demo
2. Dense Retrieval Demo
3. Knowledge Graph Demo
4. Ontology Reasoning Demo
5. Gating Mechanism Demo
6. Full Pipeline Demo
7. Run Proof-of-Concept Tests
8. Configuration
9. About
0. Exit
--------------------
```

### 2. Python API Usage

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Sample documents and query
documents = [
    "Machine learning algorithms can learn patterns from data.",
    "Deep learning uses neural networks for complex tasks.",
    "Natural language processing enables text understanding."
]

query = "How do neural networks work?"

# Build indexes
pipeline.cooccurrence.build_cooccurrence_matrix(documents)
pipeline.dense.build_index(documents)
pipeline.kg.build_knowledge_graph(documents)

# Retrieve and rank
results = pipeline.retrieve_and_rank(query, documents, top_k=3)

for doc, score in results:
    print(f"Score: {score:.3f} - {doc[:50]}...")
```

### 3. Web Application

#### Development Mode
```bash
cd app/
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

#### Production Mode
```bash
# Using Gunicorn
pip install gunicorn
cd app/
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Access at:** http://localhost:5000

---

## 🐳 Docker Deployment

### 1. Using Docker Compose (Recommended)

**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  rag-second-brain:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DENSE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-second-brain
    restart: unless-stopped
```

**Build and run:**
```bash
# Build and start services
docker-compose up --build -d

# View logs
docker-compose logs -f rag-second-brain

# Stop services
docker-compose down
```

### 2. Dockerfile

**File: `Dockerfile`**
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app/app.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app.app:app"]
```

### 3. Build and Deploy

```bash
# Build image
docker build -t rag-second-brain:latest .

# Run container
docker run -d \
  --name rag-second-brain \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e FLASK_ENV=production \
  rag-second-brain:latest

# View logs
docker logs -f rag-second-brain

# Stop container
docker stop rag-second-brain
docker rm rag-second-brain
```

---

## ☁️ Cloud Deployment

### 1. AWS Deployment

#### Using AWS ECS
```bash
# Create ECR repository
aws ecr create-repository --repository-name rag-second-brain

# Build and push to ECR
docker build -t rag-second-brain .
docker tag rag-second-brain:latest $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/rag-second-brain:latest
docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/rag-second-brain:latest

# Deploy using ECS task definition
aws ecs create-service --cluster rag-cluster --service-name rag-service --task-definition rag-task:1
```

#### ECS Task Definition (`task-definition.json`)
```json
{
    "family": "rag-second-brain",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "rag-second-brain",
            "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/rag-second-brain:latest",
            "portMappings": [
                {
                    "containerPort": 5000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "FLASK_ENV", "value": "production"},
                {"name": "DATABASE_URL", "value": "postgresql://..."}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/rag-second-brain",
                    "awslogs-region": "us-west-2",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

### 2. Google Cloud Platform

#### Using Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/rag-second-brain

# Deploy to Cloud Run
gcloud run deploy rag-second-brain \
  --image gcr.io/$PROJECT_ID/rag-second-brain \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### 3. Azure Container Instances

```bash
# Create resource group
az group create --name rag-rg --location eastus

# Create container instance
az container create \
  --resource-group rag-rg \
  --name rag-second-brain \
  --image your-registry/rag-second-brain:latest \
  --cpu 2 \
  --memory 4 \
  --ports 5000 \
  --environment-variables FLASK_ENV=production
```

---

## 🔧 Performance Optimization

### 1. Model Optimization

#### Use Smaller Models
```python
# Configuration for resource-constrained environments
OPTIMIZED_CONFIG = {
    'dense': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # 384 dim
        'index_type': 'IndexFlatIP'
    },
    'cooccurrence': {
        'window_size': 3,  # Smaller window
        'min_count': 5     # Higher threshold
    },
    'kg': {
        'entity_threshold': 0.9,  # More selective
        'max_entities': 1000      # Limit graph size
    }
}
```

#### GPU Acceleration
```python
# Enable GPU for neural components
import torch
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'

# Configure models for GPU
config = {
    'gating': {
        'device': device,
        'mixed_precision': True
    }
}
```

### 2. Memory Optimization

```python
# Efficient sparse matrix storage
from scipy.sparse import csr_matrix
import numpy as np

# Use appropriate data types
cooccurrence_matrix = csr_matrix(data, dtype=np.float32)  # Instead of float64

# Memory mapping for large indexes
import mmap
def load_large_index(path):
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return faiss.read_index_from_buffer(mm)
```

### 3. Caching Strategy

```python
from functools import lru_cache
import redis

# In-memory caching
@lru_cache(maxsize=1000)
def cached_retrieve(query: str, top_k: int):
    return pipeline.retrieve_and_rank(query, candidates, top_k)

# Redis caching for production
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_search(query, top_k):
    cache_key = f"search:{hash(query)}:{top_k}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    result = pipeline.retrieve_and_rank(query, candidates, top_k)
    redis_client.setex(cache_key, 3600, json.dumps(result))  # 1 hour cache
    return result
```

---

## 🧪 Testing Setup

### 1. Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_components.py -v
```

### 2. Integration Tests

```python
# tests/test_integration.py
import pytest
from src.pipeline import RAGPipeline

@pytest.fixture
def pipeline():
    return RAGPipeline()

@pytest.fixture 
def sample_data():
    return {
        'documents': [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "NLP helps computers understand text."
        ],
        'queries': [
            "What is machine learning?",
            "How do neural networks work?"
        ]
    }

def test_full_pipeline(pipeline, sample_data):
    # Build all indexes
    docs = sample_data['documents']
    pipeline.cooccurrence.build_cooccurrence_matrix(docs)
    pipeline.dense.build_index(docs)
    pipeline.kg.build_knowledge_graph(docs)
    
    # Test retrieval
    query = sample_data['queries'][0]
    results = pipeline.retrieve_and_rank(query, docs, top_k=2)
    
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)
    assert results[0][1] >= results[1][1]  # Properly ranked
```

### 3. Performance Tests

```python
# tests/test_performance.py
import time
import pytest
from src.pipeline import RAGPipeline

def test_retrieval_latency():
    pipeline = RAGPipeline()
    
    # Setup with realistic data size
    docs = generate_test_documents(1000)  # 1k documents
    query = "test query"
    
    # Measure end-to-end latency
    start_time = time.time()
    results = pipeline.retrieve_and_rank(query, docs, top_k=10)
    end_time = time.time()
    
    latency = end_time - start_time
    assert latency < 1.0  # Should complete within 1 second
    assert len(results) == 10

def test_memory_usage():
    pipeline = RAGPipeline()
    
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Build indexes with known data size
    docs = generate_test_documents(5000)
    pipeline.cooccurrence.build_cooccurrence_matrix(docs)
    pipeline.dense.build_index(docs)
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    # Should not exceed reasonable memory usage
    assert memory_increase < 500  # Less than 500MB for 5k docs
```

---

## 📊 Monitoring & Logging

### 1. Application Logging

**File: `src/utils/logging_config.py`**
```python
import logging
import os
from pythonjsonlogger import jsonlogger

def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE', 'logs/rag_second_brain.log')
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('rag_second_brain')
    logger.setLevel(log_level)
    
    # JSON formatter for structured logging
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger
```

### 2. Performance Monitoring

```python
# src/utils/metrics.py
import time
from functools import wraps
from typing import Dict, Any
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[f"{operation}_duration"] = duration
            del self.start_times[operation]
            return duration
    
    def record_memory(self, operation: str):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics[f"{operation}_memory_mb"] = memory_mb
        return memory_mb
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()

# Decorator for automatic timing
def monitor_performance(operation_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = getattr(wrapper, '_monitor', PerformanceMonitor())
            
            monitor.start_timer(operation_name)
            result = func(*args, **kwargs)
            monitor.end_timer(operation_name)
            monitor.record_memory(operation_name)
            
            return result
        return wrapper
    return decorator
```

---

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   RuntimeError: CUDA out of memory
   
   Solutions:
   - Reduce batch size: BATCH_SIZE=16
   - Use CPU: DEVICE=cpu
   - Use smaller models: MODEL_NAME=all-MiniLM-L6-v2
   ```

2. **Java Not Found (Pellet Reasoner)**
   ```bash
   Java not found. Please install Java.
   
   Solutions:
   # Ubuntu/Debian
   sudo apt-get install default-jdk
   
   # macOS
   brew install openjdk
   
   # Set JAVA_HOME if needed
   export JAVA_HOME=/usr/lib/jvm/default-java
   ```

3. **Memory Issues with Large Corpora**
   ```bash
   MemoryError: Unable to allocate memory
   
   Solutions:
   - Process in batches
   - Increase system memory
   - Use sparse matrices: scipy.sparse
   - Enable swap if on Linux
   ```

4. **Slow Model Downloads**
   ```bash
   # Pre-download models
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   python -m spacy download en_core_web_sm
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export FLASK_ENV=development

# Run with verbose output
python main.py --verbose

# Profile performance
python -m cProfile -o profile.stats main.py
```

---

## 🚀 Next Steps

After successful setup:

1. **Explore the Demo** - Run `python main.py` and try all menu options
2. **Read the Documentation** - Check other wiki pages for detailed API reference
3. **Experiment** - Use Jupyter notebooks in `notebooks/` directory
4. **Customize** - Modify configuration for your specific use case
5. **Deploy** - Follow production deployment guidelines above

**Useful Commands Summary:**
```bash
# Development
python main.py                    # Run demo
flask run                        # Start web app
pytest tests/ -v                 # Run tests

# Docker
docker-compose up --build        # Deploy with Docker
docker logs -f rag-second-brain  # View logs

# Production
gunicorn app:app                 # Production server
systemctl status rag-service    # Check service status
```

---

**Next:** [🔧 Configuration](CONFIGURATION.md) | **Previous:** [🌊 Data Flow](DATA_FLOW.md)