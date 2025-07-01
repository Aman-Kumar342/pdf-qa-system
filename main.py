# main.py
from pydoc import text
import PyPDF2
import gradio as gr
from sqlalchemy import Extract
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

class DocumentProcessor:
    def __init__(self):
        self.text_chunks = []
        self.chunk_size = 500  # characters per chunk
        
    def extract_text_from_pdf(self, pdf_file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
            print(f"Extracted text from PDF: {len(text)} characters")
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        # Simple sentence-based splitting
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        print(f"Text split into {len(chunks)} chunks")
        return chunks

# Add this after the DocumentProcessor class

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.chunks = []
        
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            return False
        
    def generate_embeddings(self, text_chunks: List[str]):
        """Generate embeddings for text chunks"""
        if self.model is None:
            print("Embedding model not loaded. Loading now...")
            if not self.load_embedding_model():
                return False
                
        print(f"Generating embeddings for {len(text_chunks)} chunks...")
        try:
            self.chunks = text_chunks
            self.embeddings = self.model.encode(text_chunks)
            print(f"Embeddings generated successfully! Shape: {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return False
        
    def save_embeddings(self, filepath="embeddings.pkl"):
        """Save embeddings to file"""
        if self.embeddings is None:
            print("No embeddings to save!")
            return False
            
        try:
            data = {
                'embeddings': self.embeddings,
                'chunks': self.chunks,
                'model_name': self.model_name
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"Embeddings saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
            return False
        
    def load_embeddings(self, filepath="embeddings.pkl"):
        """Load embeddings from file"""
        if not os.path.exists(filepath):
            print(f"Embeddings file {filepath} not found")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data['embeddings']
            self.chunks = data['chunks']
            self.model_name = data.get('model_name', self.model_name)
            
            print(f"Embeddings loaded from {filepath}")
            print(f"Loaded {len(self.chunks)} chunks with embeddings shape: {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False

# Test the Embedding Manager
print("\n" + "="*50)
print("Testing Embedding Manager...")
embedding_manager = EmbeddingManager()
print("Embedding Manager initialized!")

# Test the Document Processor
print("\n" + "="*50)
print("Testing Document Processor...")
doc_processor = DocumentProcessor()
print("Document Processor initialized successfully!")

# Test with sample text (since we don't have a PDF yet)
print("\n" + "="*50)
print("Testing text chunking...")

sample_text = """
This is a sample document for testing. It contains multiple sentences that will be split into chunks.
The document processor should handle this text properly. Each chunk should be of reasonable size.
This helps in better processing and retrieval of information. The system will use these chunks for embedding generation.
Natural language processing is fascinating. It allows computers to understand human language.
Machine learning models can process text effectively. This enables building intelligent applications.
"""

chunks = doc_processor.split_text_into_chunks(sample_text)
print(f"Sample text chunked into {len(chunks)} pieces:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:100]}...")

# Test embedding generation with our sample chunks
print("\n" + "="*50)
print("Testing embedding generation...")

# Use the chunks we created in Step 2
if len(chunks) > 0:
    print("Loading embedding model...")
    if embedding_manager.load_embedding_model():
        print("Generating embeddings for sample chunks...")
        if embedding_manager.generate_embeddings(chunks):
            print("✅ Embeddings generated successfully!")
            
            # Test saving embeddings
            if embedding_manager.save_embeddings("test_embeddings.pkl"):
                print("✅ Embeddings saved successfully!")
                
                # Test loading embeddings
                new_embedding_manager = EmbeddingManager()
                if new_embedding_manager.load_embeddings("test_embeddings.pkl"):
                    print("✅ Embeddings loaded successfully!")
                else:
                    print("❌ Failed to load embeddings")
            else:
                print("❌ Failed to save embeddings")
        else:
            print("❌ Failed to generate embeddings")
    else:
        print("❌ Failed to load embedding model")
else:
    print("❌ No chunks available for testing")

