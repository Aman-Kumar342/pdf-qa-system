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

# Add this after the EmbeddingManager class

class VectorDatabase:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = None
        
    def build_index(self, embeddings, chunks):
        """Build FAISS index from embeddings"""
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings provided to build index")
            return False
            
        try:
            print("Building FAISS index...")
            self.dimension = embeddings.shape[1]
            
            # Create FAISS index (Inner Product for cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            # Add embeddings to index
            self.index.add(embeddings_normalized.astype('float32'))
            self.chunks = chunks
            
            print(f"FAISS index built successfully!")
            print(f"Index contains {self.index.ntotal} vectors of dimension {self.dimension}")
            return True
            
        except Exception as e:
            print(f"Error building FAISS index: {str(e)}")
            return False
        
    def search(self, query_embedding, k=3):
        """Search for most similar chunks"""
        if self.index is None:
            print("No index built yet!")
            return []
            
        if k > len(self.chunks):
            k = len(self.chunks)
            
        try:
            # Prepare query embedding
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):  # Safety check
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(score),
                        'index': int(idx),
                        'rank': i + 1
                    })
            
            print(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            print(f"Error searching index: {str(e)}")
            return []
    
    def save_index(self, filepath="vector_index.faiss"):
        """Save FAISS index to file"""
        if self.index is None:
            print("No index to save!")
            return False
            
        try:
            faiss.write_index(self.index, filepath)
            
            # Save chunks separately
            chunks_filepath = filepath.replace('.faiss', '_chunks.pkl')
            with open(chunks_filepath, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'dimension': self.dimension
                }, f)
                
            print(f"Index saved to {filepath}")
            print(f"Chunks saved to {chunks_filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, filepath="vector_index.faiss"):
        """Load FAISS index from file"""
        chunks_filepath = filepath.replace('.faiss', '_chunks.pkl')
        
        if not os.path.exists(filepath) or not os.path.exists(chunks_filepath):
            print(f"Index files not found: {filepath} or {chunks_filepath}")
            return False
            
        try:
            # Load index
            self.index = faiss.read_index(filepath)
            
            # Load chunks
            with open(chunks_filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data['chunks']
            self.dimension = data['dimension']
            
            print(f"Index loaded from {filepath}")
            print(f"Loaded {len(self.chunks)} chunks, index contains {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False


# Alternative lightweight models (add this as a comment in your code)
# For low memory systems, you can use:
# - "gpt2" (smaller, faster)
# - "microsoft/DialoGPT-small" (current choice)
# - "distilgpt2" (even smaller)

# To change model, modify this line in ModelLoader.__init__:
# self.model_name = "gpt2"  # or any other model


# Add this after the VectorDatabase class



class ModelLoader:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the language model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Added padding token")
            
            # Load model
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Move model to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def generate_answer(self, context: str, question: str, max_length=150) -> str:
        """Generate answer based on context and question"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please load the model first."
        
        try:
            # Create prompt
            prompt = f"Context: {context[:1000]}\n\nQuestion: {question}\n\nAnswer:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response[len(prompt):].strip()
            
            # Clean up the answer
            answer = answer.split('\n')[0].strip()  # Take first line only
            
            return answer if answer else "I couldn't generate a proper answer."
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def test_generation(self):
        """Test the model with a simple prompt"""
        if self.model is None:
            return "Model not loaded"
        
        test_context = "Python is a programming language. It is easy to learn and powerful."
        test_question = "What is Python?"
        
        answer = self.generate_answer(test_context, test_question)
        return answer

# Test the Model Loader
print("\n" + "="*50)
print("Testing Model Loader...")
model_loader = ModelLoader()
print("Model Loader initialized!")


# Test the Vector Database
print("\n" + "="*50)
print("Testing Vector Database...")
vector_db = VectorDatabase()
print("Vector Database initialized!")


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


# Test vector database with existing embeddings
print("\n" + "="*50)
print("Testing Vector Database operations...")

if embedding_manager.embeddings is not None and len(embedding_manager.chunks) > 0:
    # Build index
    if vector_db.build_index(embedding_manager.embeddings, embedding_manager.chunks):
        print("✅ Vector index built successfully!")
        
        # Test search functionality
        print("\nTesting search functionality...")
        
        # Create a test query (use first chunk as query for testing)
        test_query = "sample document testing"
        print(f"Test query: '{test_query}'")
        
        # Generate embedding for test query
        query_embedding = embedding_manager.model.encode([test_query])[0]
        
        # Search for similar chunks
        results = vector_db.search(query_embedding, k=2)
        
        if results:
            print("✅ Search completed successfully!")
            for result in results:
                print(f"Rank {result['rank']}: Score {result['score']:.4f}")
                print(f"Chunk: {result['chunk'][:100]}...")
                print("-" * 50)
        else:
            print("❌ No search results found")
            
        # Test saving and loading index
        print("\nTesting save/load functionality...")
        if vector_db.save_index("test_vector_index.faiss"):
            print("✅ Index saved successfully!")
            
            # Test loading
            new_vector_db = VectorDatabase()
            if new_vector_db.load_index("test_vector_index.faiss"):
                print("✅ Index loaded successfully!")
                
                # Test search on loaded index
                test_results = new_vector_db.search(query_embedding, k=1)
                if test_results:
                    print("✅ Search on loaded index works!")
                else:
                    print("❌ Search on loaded index failed")
            else:
                print("❌ Failed to load index")
        else:
            print("❌ Failed to save index")
    else:
        print("❌ Failed to build vector index")
else:
    print("❌ No embeddings available for testing")


# Test model loading and generation
print("\n" + "="*50)
print("Testing Model Loading and Generation...")

print("⚠️  Note: This will download the model (about 500MB) on first run...")
print("Loading language model...")

if model_loader.load_model():
    print("✅ Model loaded successfully!")
    
    # Test basic generation
    print("\nTesting basic text generation...")
    test_answer = model_loader.test_generation()
    print(f"Test Answer: {test_answer}")
    
    # Test with our sample data
    if len(embedding_manager.chunks) > 0:
        print("\nTesting with sample context...")
        sample_context = embedding_manager.chunks[0]
        sample_question = "What is this document about?"
        
        answer = model_loader.generate_answer(sample_context, sample_question)
        print(f"Question: {sample_question}")
        print(f"Context: {sample_context[:100]}...")
        print(f"Answer: {answer}")
        print("✅ Context-based generation works!")
    else:
        print("❌ No sample chunks available for testing")
        
else:
    print("❌ Failed to load model")
    print("💡 Tip: Try using a smaller model like 'gpt2' if you have memory issues")
