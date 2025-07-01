# System Requirements & Dependencies

## Hardware Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: Dual-core processor
- **Internet**: Required for initial model downloads

### Recommended Requirements
- **RAM**: 8GB or higher
- **Storage**: 5GB free space
- **CPU**: Quad-core processor or better
- **GPU**: CUDA-compatible GPU (optional, for faster processing)

## Software Requirements

### Operating System
- Windows 10/11
- macOS 10.14+
- Ubuntu 18.04+ or equivalent Linux distribution

### Python Environment
- Python 3.8 or higher
- pip package manager
- Virtual environment support

## Dependencies

### Core Dependencies

PyPDF2==3.0.1 # PDF text extraction
gradio==3.50.0 # Web interface
transformers==4.30.0 # Language models
sentence-transformers==2.2.2 # Text embeddings
torch==2.0.1 # PyTorch framework
faiss-cpu==1.7.4 # Vector similarity search
numpy==1.24.3 # Numerical computing


### Additional Dependencies
accelerate==0.20.3 # Model acceleration
langchain==0.0.350 # LLM framework utilities
huggingface_hub==0.16.4 # Model hub integration
psutil # System monitoring
scikit-learn # Machine learning utilities
nltk # Natural language processing


## Model Downloads

The system automatically downloads these models on first run:
- **Embedding Model**: all-MiniLM-L6-v2 (~90MB)
- **Language Model**: microsoft/DialoGPT-small (~500MB)

Total download size: ~600MB
