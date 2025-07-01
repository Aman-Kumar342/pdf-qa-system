# 📚 PDF Question-Answering System

An intelligent PDF document analysis system that allows users to upload PDF documents and ask natural language questions about their content. Built with Python, Transformers, FAISS, and Gradio.

## 🌟 Features

- **PDF Text Extraction**: Automatically extracts and processes text from PDF documents
- **Intelligent Chunking**: Splits documents into semantic chunks for better retrieval
- **Vector Search**: Uses FAISS for fast similarity search across document content
- **AI-Powered Answers**: Generates contextual answers using pre-trained language models
- **Interactive Web Interface**: User-friendly Gradio interface with chat functionality
- **Document Analytics**: Provides statistics and insights about processed documents
- **Export Functionality**: Save conversations and analysis results
- **Performance Monitoring**: Track system performance and resource usage

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **AI/ML**: Transformers, Sentence-Transformers, PyTorch
- **Vector Database**: FAISS
- **PDF Processing**: PyPDF2
- **Web Interface**: Gradio
- **Data Processing**: NumPy, Scikit-learn

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB recommended)
- 2GB+ free disk space
- Internet connection (for model downloads)

## 🚀 Quick Start

### 1. Clone the Repository
git clone <your-repo-url>
cd pdf-qa-system


### 2. Create Virtual Environment
python -m venv pdf_qa_env

Windows
pdf_qa_env\Scripts\activate

Mac/Linux
source pdf_qa_env/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Application
python deploy.py

### 5. Open in Browser
Navigate to `http://127.0.0.1:7860` in your web browser.

## 📖 Usage Guide

### Basic Usage
1. **Initialize System**: Click "Initialize System" to load AI models
2. **Upload PDF**: Select and upload your PDF document
3. **Process Document**: Click "Process PDF" to analyze the document
4. **Ask Questions**: Type questions about the document content
5. **Get Answers**: Receive AI-generated answers with source citations

### Advanced Features
- **Document Statistics**: View detailed analytics about your document
- **Export Conversations**: Save your Q&A sessions as markdown files
- **Performance Monitoring**: Track system performance metrics
- **Multiple Interface Options**: Choose between basic and enhanced UI

## 🏗️ Project Structure

pdf-qa-system/
├── main.py # Main application code
├── config.py # Configuration settings
├── deploy.py # Production deployment script
├── test_system.py # Test suite
├── launch.py # Simple launcher
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
├── README.md # This file
├── embeddings/ # Stored embeddings (auto-created)
├── models/ # Cached models (auto-created)
├── exports/ # Exported files (auto-created)
└── pdf_qa_env/ # Virtual environment (ignored by git)

## 🧪 Testing

Run the comprehensive test suite:
python test_system.py

This will test all system components:
- Document processing
- Embedding generation
- Vector database operations
- Language model functionality
- Complete system integration

## ⚙️ Configuration

Edit `config.py` to customize:
- Model selection (embedding and language models)
- Processing parameters (chunk size, context length)
- Server settings (host, port, debug mode)
- File paths and directories

## 🔧 Troubleshooting

### Common Issues

**Memory Errors**
- Reduce chunk size in `config.py`
- Use smaller models (e.g., "gpt2" instead of "microsoft/DialoGPT-small")
- Ensure 4GB+ RAM available

**Model Loading Errors**
- Check internet connection for model downloads
- Verify disk space (models can be 500MB+)
- Try clearing model cache and re-downloading

**PDF Processing Errors**
- Ensure PDF is text-based (not scanned images)
- Check file size limits (50MB default)
- Verify PDF is not password-protected

### Performance Optimization

**For Better Speed**
- Use GPU if available (CUDA)
- Reduce `MAX_CHUNKS_FOR_CONTEXT` in config
- Use smaller embedding models

**For Better Accuracy**
- Increase chunk overlap
- Use larger language models
- Adjust similarity thresholds

## 📊 Performance Metrics

The system tracks:
- Document processing times
- Query response times
- Memory and CPU usage
- Vector search performance

Access metrics through the "Document Analysis" tab in the enhanced interface.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Facebook Research** for FAISS vector search
- **Gradio** for the web interface framework
- **PyPDF2** for PDF processing capabilities

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the test suite to identify problems
3. Check system requirements and dependencies
4. Create an issue with detailed error logs

## 🔮 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT, etc.)
- [ ] Multi-language document support
- [ ] Advanced document preprocessing
- [ ] Integration with cloud storage services
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Advanced analytics and visualizations

---

**Built with ❤️ using Python and AI**

