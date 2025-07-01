# Installation Guide

## Step-by-Step Installation

### 1. System Preparation

#### Windows
Install Python from python.org
Ensure Python is added to PATH
python --version # Should show 3.8+

#### macOS
nstall Python using Homebrew
brew install python
python3 --version


#### Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version

### 2. Project Setup
Clone or download the project
git clone <repository-url>
cd pdf-qa-system

Create virtual environment
python -m venv pdf_qa_env

Activate virtual environment
Windows:
pdf_qa_env\Scripts\activate

macOS/Linux:
source pdf_qa_env/bin/activate


### 3. Install Dependencies

Install all required packages
pip install -r requirements.txt

Verify installation
python -c "import torch; print('PyTorch:', torch.version)"
python -c "import gradio; print('Gradio:', gradio.version)"

### 4. Initial Setup
Run system tests
python test_system.py

If tests pass, run the application
python deploy.py

### 5. First Run

1. Open browser to `http://127.0.0.1:7860`
2. Click "Initialize System" (will download models)
3. Wait for "System initialized successfully" message
4. Upload a test PDF and try asking questions

## Troubleshooting Installation

### Common Issues

**Python Version Error**
Check Python version
python --version

If < 3.8, install newer Python version

**pip Installation Fails**
Upgrade pip
python -m pip install --upgrade pip

Try installing with --no-cache-dir
pip install --no-cache-dir -r requirements.txt

**Virtual Environment Issues**
Remove and recreate virtual environment
rm -rf pdf_qa_env # Linux/macOS
rmdir /s pdf_qa_env # Windows
python -m venv pdf_qa_env

**Model Download Fails**
Check internet connection
Clear Hugging Face cache
rm -rf ~/.cache/huggingface/ # Linux/macOS

Retry initialization

### Platform-Specific Notes

#### Windows
- Use Command Prompt or PowerShell
- May need to install Microsoft Visual C++ Redistributable
- Windows Defender might flag model downloads

#### macOS
- May need to install Xcode Command Line Tools
- Use `python3` instead of `python` if needed

#### Linux
- May need to install additional system packages:
sudo apt install build-essential python3-dev

## Verification

After installation, verify everything works:

Run comprehensive tests
python test_system.py

Should show all tests passing
If any tests fail, check the error messages
undefined

