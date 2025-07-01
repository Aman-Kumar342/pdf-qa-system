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

# Add this after the ModelLoader class

class PDFQASystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = VectorDatabase()
        self.model_loader = ModelLoader()
        self.is_ready = False
        self.pdf_processed = False
        
    def setup(self):
        """Initialize all components"""
        print("Setting up PDF QA System...")
        print("This may take a few minutes on first run...")
        
        try:
            # Load embedding model
            print("1/2 Loading embedding model...")
            if not self.embedding_manager.load_embedding_model():
                return False
            
            # Load language model
            print("2/2 Loading language model...")
            if not self.model_loader.load_model():
                return False
            
            self.is_ready = True
            print("✅ PDF QA System ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up system: {str(e)}")
            return False
        
    def process_pdf(self, pdf_file_path):
        """Process uploaded PDF"""
        if not self.is_ready:
            return "❌ System not ready. Please run setup() first."
        
        try:
            print(f"Processing PDF: {pdf_file_path}")
            
            # Check if file exists
            if not os.path.exists(pdf_file_path):
                return f"❌ File not found: {pdf_file_path}"
            
            # Extract text from PDF
            print("Extracting text from PDF...")
            text = self.doc_processor.extract_text_from_pdf(pdf_file_path)
            
            if not text.strip():
                return "❌ No text could be extracted from the PDF"
            
            # Split into chunks
            print("Splitting text into chunks...")
            chunks = self.doc_processor.split_text_into_chunks(text)
            
            if not chunks:
                return "❌ No chunks created from the text"
            
            # Generate embeddings
            print("Generating embeddings...")
            if not self.embedding_manager.generate_embeddings(chunks):
                return "❌ Failed to generate embeddings"
            
            # Build vector database
            print("Building vector database...")
            if not self.vector_db.build_index(self.embedding_manager.embeddings, chunks):
                return "❌ Failed to build vector database"
            
            self.pdf_processed = True
            return f"✅ PDF processed successfully! Created {len(chunks)} chunks for analysis."
            
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}"
    
    def answer_question(self, question: str, top_k=3) -> str:
        """Answer question based on processed PDF"""
        if not self.is_ready:
            return "❌ System not ready. Please run setup() first."
            
        if not self.pdf_processed:
            return "❌ No PDF processed yet. Please upload and process a PDF first."
        
        if not question.strip():
            return "❌ Please provide a valid question."
        
        try:
            print(f"Answering question: {question}")
            
            # Generate query embedding
            print("Generating query embedding...")
            query_embedding = self.embedding_manager.model.encode([question])[0]
            
            # Search for relevant chunks
            print("Searching for relevant information...")
            results = self.vector_db.search(query_embedding, k=top_k)
            
            if not results:
                return "❌ No relevant information found in the document."
            
            # Combine top results as context
            context_parts = []
            for i, result in enumerate(results):
                context_parts.append(f"[Context {i+1}]: {result['chunk']}")
            
            context = "\n\n".join(context_parts)
            
            # Generate answer
            print("Generating answer...")
            answer = self.model_loader.generate_answer(context, question)
            
            # Format response with sources
            response = f"**Answer:** {answer}\n\n"
            response += f"**Sources used ({len(results)} most relevant chunks):**\n"
            
            for i, result in enumerate(results):
                response += f"{i+1}. Score: {result['score']:.3f} - {result['chunk'][:100]}...\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def get_system_status(self):
        """Get current system status"""
        status = {
            'embedding_model_loaded': self.embedding_manager.model is not None,
            'language_model_loaded': self.model_loader.model is not None,
            'system_ready': self.is_ready,
            'pdf_processed': self.pdf_processed,
            'chunks_count': len(self.embedding_manager.chunks) if self.embedding_manager.chunks else 0
        }
        return status

# Add this after the PDFQASystem class
class GradioInterface:
    def __init__(self):
        self.qa_system = PDFQASystem()
        self.setup_complete = False
        
    def initialize_system(self):
        """Initialize the QA system"""
        try:
            success = self.qa_system.setup()
            if success:
                self.setup_complete = True
                return "✅ System initialized successfully! You can now upload a PDF."
            else:
                return "❌ Failed to initialize system. Please check the logs."
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"
    
    def upload_pdf(self, pdf_file):
        """Handle PDF upload and processing"""
        if not self.setup_complete:
            return "❌ Please initialize the system first.", ""
        
        if pdf_file is None:
            return "❌ Please upload a PDF file.", ""
        
        try:
            # Process the uploaded PDF
            result = self.qa_system.process_pdf(pdf_file.name)
            
            if "✅" in result:
                # Get document preview
                if self.qa_system.embedding_manager.chunks:
                    preview = "📄 **Document Preview:**\n\n"
                    preview += f"Total chunks: {len(self.qa_system.embedding_manager.chunks)}\n\n"
                    preview += "**First few chunks:**\n"
                    for i, chunk in enumerate(self.qa_system.embedding_manager.chunks[:3]):
                        preview += f"\n**Chunk {i+1}:** {chunk[:200]}...\n"
                    return result, preview
                else:
                    return result, "No preview available."
            else:
                return result, ""
                
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}", ""
    
    def answer_question(self, question, history):
        """Handle question answering"""
        if not self.setup_complete:
            return history + [("System not ready", "❌ Please initialize the system first.")]
        
        if not self.qa_system.pdf_processed:
            return history + [(question, "❌ Please upload and process a PDF first.")]
        
        if not question.strip():
            return history + [("", "❌ Please enter a valid question.")]
        
        try:
            # Get answer from QA system
            answer = self.qa_system.answer_question(question)
            
            # Add to chat history
            history.append((question, answer))
            return history
            
        except Exception as e:
            error_msg = f"❌ Error answering question: {str(e)}"
            history.append((question, error_msg))
            return history
    
    def get_system_info(self):
        """Get current system status information"""
        status = self.qa_system.get_system_status()
        
        info = "🔧 **System Status:**\n\n"
        info += f"• Embedding Model: {'✅ Loaded' if status['embedding_model_loaded'] else '❌ Not loaded'}\n"
        info += f"• Language Model: {'✅ Loaded' if status['language_model_loaded'] else '❌ Not loaded'}\n"
        info += f"• System Ready: {'✅ Yes' if status['system_ready'] else '❌ No'}\n"
        info += f"• PDF Processed: {'✅ Yes' if status['pdf_processed'] else '❌ No'}\n"
        info += f"• Document Chunks: {status['chunks_count']}\n"
        
        return info
    
    # 👇 NEW METHODS FROM STEP 8.2 - ADD THESE
    def export_chat(self, history):
        """Export chat history"""
        if not history:
            return "No conversation to export."
        
        export_text = "# PDF Q&A Chat Export\n\n"
        for i, (question, answer) in enumerate(history, 1):
            export_text += f"## Exchange {i}\n"
            export_text += f"**Question:** {question}\n\n"
            export_text += f"**Answer:** {answer}\n\n"
            export_text += "---\n\n"
        
        filename = f"chat_export_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(export_text)
        
        return f"✅ Chat exported to {filename}"

    def get_document_stats(self):
        """Get document statistics"""
        if not self.qa_system.pdf_processed:
            return "No document processed."
        
        chunks = self.qa_system.embedding_manager.chunks
        total_chars = sum(len(chunk) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        stats = f"""
📊 **Document Statistics:**

• Total Chunks: {len(chunks)}
• Total Characters: {total_chars:,}
• Average Chunk Size: {avg_chunk_size:.0f} characters
• Embedding Dimension: {self.qa_system.embedding_manager.embeddings.shape[1] if self.qa_system.embedding_manager.embeddings is not None else 'N/A'}
• Vector Database Size: {self.qa_system.vector_db.index.ntotal if self.qa_system.vector_db.index else 0} vectors
        """
        
        return stats

    def create_enhanced_interface(self):
        """Create enhanced Gradio interface with additional features"""
        
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .feature-box {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        """
        
        with gr.Blocks(css=css, title="Enhanced PDF Q&A System") as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>📚 Enhanced PDF Question-Answering System</h1>
                <p>Upload, analyze, and interact with your PDF documents using AI!</p>
            </div>
            """)
            
            with gr.Tabs():
                # Main Q&A Tab
                with gr.TabItem("🔍 Question & Answer"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## 🚀 Setup")
                            init_btn = gr.Button("Initialize System", variant="primary")
                            init_output = gr.Textbox(label="Status", interactive=False)
                            
                            gr.Markdown("## 📄 Upload Document")
                            pdf_input = gr.File(label="PDF Document", file_types=[".pdf"])
                            upload_btn = gr.Button("Process PDF", variant="secondary")
                            upload_output = gr.Textbox(label="Processing Status", interactive=False)
                            
                        with gr.Column(scale=2):
                            gr.Markdown("## 💬 Chat with Document")
                            chatbot = gr.Chatbot(height=400, show_label=True)
                            
                            with gr.Row():
                                question_input = gr.Textbox(
                                    label="Your Question",
                                    placeholder="Ask anything about the document...",
                                    scale=4
                                )
                                ask_btn = gr.Button("Ask", variant="primary", scale=1)
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")
                                export_btn = gr.Button("Export Chat")
                            
                            export_status = gr.Textbox(label="Export Status", interactive=False)
                
                # Document Analysis Tab
                with gr.TabItem("📊 Document Analysis"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## 📈 Document Statistics")
                            stats_btn = gr.Button("Get Statistics")
                            stats_output = gr.Markdown()
                            
                            gr.Markdown("## 💡 Suggested Questions")
                            suggestions_btn = gr.Button("Get Question Suggestions")
                            suggestions_output = gr.Markdown()
                        
                        with gr.Column():
                            gr.Markdown("## 📋 Document Preview")
                            preview_output = gr.Markdown()
                            
                            gr.Markdown("## 🔧 System Status")
                            status_btn = gr.Button("Check System Status")
                            status_output = gr.Markdown()
            
            # Event handlers for main tab
            init_btn.click(fn=self.initialize_system, outputs=init_output)
            upload_btn.click(fn=self.upload_pdf, inputs=pdf_input, outputs=[upload_output, preview_output])
            ask_btn.click(fn=self.answer_question, inputs=[question_input, chatbot], outputs=chatbot).then(lambda: "", outputs=question_input)
            question_input.submit(fn=self.answer_question, inputs=[question_input, chatbot], outputs=chatbot).then(lambda: "", outputs=question_input)
            clear_btn.click(lambda: [], outputs=chatbot)
            export_btn.click(fn=self.export_chat, inputs=chatbot, outputs=export_status)
            
            # Event handlers for analysis tab
            stats_btn.click(fn=self.get_document_stats, outputs=stats_output)
            status_btn.click(fn=self.get_system_info, outputs=status_output)
            suggestions_btn.click(
                fn=lambda: "💡 **Suggested Questions:**\n\n" + "\n".join([f"• {q}" for q in [
                    "What is the main topic of this document?",
                    "What are the key findings or conclusions?",
                    "Who are the main authors mentioned?",
                    "What methodology is described?",
                    "What are the important numbers or dates?",
                    "What recommendations are provided?"
                ]]),
                outputs=suggestions_output
            )
        
        return interface
    
    def create_interface(self):
        """Create and configure Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
        
        with gr.Blocks(css=css, title="PDF Question-Answering System") as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>📚 PDF Question-Answering System</h1>
                <p>Upload a PDF document and ask questions about its content!</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # System initialization
                    gr.Markdown("## 🚀 Step 1: Initialize System")
                    init_btn = gr.Button("Initialize System", variant="primary")
                    init_output = gr.Textbox(label="Initialization Status", interactive=False)
                    
                    # PDF Upload
                    gr.Markdown("## 📄 Step 2: Upload PDF")
                    pdf_input = gr.File(
                        label="Upload PDF Document", 
                        file_types=[".pdf"]
                    )
                    upload_btn = gr.Button("Process PDF", variant="secondary")
                    upload_output = gr.Textbox(label="Processing Status", interactive=False)
                    
                    # System Status
                    gr.Markdown("## ℹ️ System Information")
                    status_btn = gr.Button("Check Status")
                    status_output = gr.Markdown()
                
                with gr.Column(scale=2):
                    # Document Preview
                    gr.Markdown("## 👀 Document Preview")
                    preview_output = gr.Markdown()
                    
                    # Question-Answer Interface
                    gr.Markdown("## ❓ Ask Questions")
                    chatbot = gr.Chatbot(
                        label="Q&A Chat",
                        height=400,
                        show_label=True
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about the uploaded document...",
                            scale=4
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)
                    
                    # Clear chat button
                    clear_btn = gr.Button("Clear Chat", variant="stop")
            
            # Example questions
            gr.Markdown("""
            ## 💡 Example Questions You Can Ask:
            - What is the main topic of this document?
            - Can you summarize the key points?
            - What are the conclusions mentioned?
            - Who are the authors mentioned?
            - What methodology was used?
            """)
            
            # Event handlers
            init_btn.click(
                fn=self.initialize_system,
                outputs=init_output
            )
            
            upload_btn.click(
                fn=self.upload_pdf,
                inputs=pdf_input,
                outputs=[upload_output, preview_output]
            )
            
            ask_btn.click(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",  # Clear the input box
                outputs=question_input
            )
            
            question_input.submit(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",  # Clear the input box
                outputs=question_input
            )
            
            clear_btn.click(
                lambda: [],
                outputs=chatbot
            )
            
            status_btn.click(
                fn=self.get_system_info,
                outputs=status_output
            )
        
        return interface

# Add this after the GradioInterface class

import time
import psutil
import threading

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'query_response_times': [],
            'system_load': []
        }
        self.monitoring = False
        
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_system)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        
    def _monitor_system(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # Get system metrics
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent()
                
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['system_load'].append(cpu_percent)
                
                time.sleep(5)  # Monitor every 5 seconds
            except:
                pass
                
    def log_processing_time(self, operation, duration):
        """Log processing time for operations"""
        self.metrics['processing_times'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        })
        
    def log_query_time(self, duration):
        """Log query response time"""
        self.metrics['query_response_times'].append(duration)
        
    def get_performance_report(self):
        """Generate performance report"""
        report = "📈 **Performance Report**\n\n"
        
        # Processing times
        if self.metrics['processing_times']:
            avg_processing = sum(p['duration'] for p in self.metrics['processing_times']) / len(self.metrics['processing_times'])
            report += f"• Average Processing Time: {avg_processing:.2f} seconds\n"
            
        # Query response times
        if self.metrics['query_response_times']:
            avg_query = sum(self.metrics['query_response_times']) / len(self.metrics['query_response_times'])
            report += f"• Average Query Response: {avg_query:.2f} seconds\n"
            
        # Memory usage
        if self.metrics['memory_usage']:
            avg_memory = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
            max_memory = max(self.metrics['memory_usage'])
            report += f"• Average Memory Usage: {avg_memory:.1f}%\n"
            report += f"• Peak Memory Usage: {max_memory:.1f}%\n"
            
        # System load
        if self.metrics['system_load']:
            avg_cpu = sum(self.metrics['system_load']) / len(self.metrics['system_load'])
            report += f"• Average CPU Usage: {avg_cpu:.1f}%\n"
            
        return report


# Add this after the GradioInterface class

class AdvancedFeatures:
    def __init__(self, qa_system):
        self.qa_system = qa_system
        self.conversation_history = []
        
    def save_conversation(self, question, answer):
        """Save conversation for context"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def export_conversation(self):
        """Export conversation history"""
        if not self.conversation_history:
            return "No conversation to export."
        
        export_text = "# PDF Q&A Conversation History\n\n"
        for i, conv in enumerate(self.conversation_history, 1):
            export_text += f"## Question {i} ({conv['timestamp']})\n"
            export_text += f"**Q:** {conv['question']}\n\n"
            export_text += f"**A:** {conv['answer']}\n\n"
            export_text += "---\n\n"
        
        # Save to file
        filename = f"conversation_export_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(export_text)
        
        return f"✅ Conversation exported to {filename}"
    
    def get_document_summary(self):
        """Generate document summary"""
        if not self.qa_system.pdf_processed:
            return "No document processed yet."
        
        # Use first few chunks to create summary
        chunks = self.qa_system.embedding_manager.chunks[:5]
        combined_text = " ".join(chunks)
        
        summary_question = "What is the main topic and key points of this document?"
        summary = self.qa_system.answer_question(summary_question)
        
        return f"📋 **Document Summary:**\n\n{summary}"
    
    def suggest_questions(self):
        """Suggest relevant questions based on document content"""
        if not self.qa_system.pdf_processed:
            return []
        
        # Analyze document content and suggest questions
        suggestions = [
            "What is the main topic of this document?",
            "What are the key findings or conclusions?",
            "Who are the main authors or contributors mentioned?",
            "What methodology or approach is described?",
            "What are the important dates or numbers mentioned?",
            "What recommendations are provided?",
            "What are the limitations or challenges discussed?",
            "What future work or next steps are suggested?"
        ]
        
        return suggestions


# Create and launch the Gradio interface
print("\n" + "="*50)
print("Creating Gradio Interface...")
gradio_app = GradioInterface()
interface = gradio_app.create_interface()
print("Gradio Interface created successfully!")


# Initialize the complete system
print("\n" + "="*50)
print("Initializing Complete PDF QA System...")
qa_system = PDFQASystem()
print("PDF QA System created!")


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

# Test the complete PDF QA System
print("\n" + "="*50)
print("Testing Complete PDF QA System...")

# Check initial status
print("Initial system status:")
status = qa_system.get_system_status()
for key, value in status.items():
    print(f"  {key}: {value}")

# Setup the system
print("\nSetting up the system...")
if qa_system.setup():
    print("✅ System setup completed!")
    
    # Check status after setup
    print("\nSystem status after setup:")
    status = qa_system.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test with sample text (simulating PDF processing)
    print("\nTesting with sample data...")
    
    # Manually add sample data to test the Q&A functionality
    sample_chunks = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. Python is widely used in machine learning.",
        "Natural language processing (NLP) is a field of AI that helps computers understand human language. Libraries like NLTK and spaCy are popular in Python."
    ]
    
    # Process sample data
    if qa_system.embedding_manager.generate_embeddings(sample_chunks):
        if qa_system.vector_db.build_index(qa_system.embedding_manager.embeddings, sample_chunks):
            qa_system.pdf_processed = True
            print("✅ Sample data processed successfully!")
            
            # Test questions
            test_questions = [
                "What is Python?",
                "Who created Python?",
                "What is machine learning?",
                "What libraries are used for NLP in Python?"
            ]
            
            print("\nTesting Q&A functionality:")
            for question in test_questions:
                print(f"\n{'='*30}")
                print(f"Question: {question}")
                print("="*30)
                answer = qa_system.answer_question(question)
                print(answer)
        else:
            print("❌ Failed to build vector database with sample data")
    else:
        print("❌ Failed to generate embeddings for sample data")
        
else:
    print("❌ System setup failed")

print("\n" + "="*50)
print("PDF QA System testing completed!")
print("Next step: Create Gradio interface for user interaction")


# Launch the application
# Replace the launch code at the end of main.py
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Launching Enhanced PDF Question-Answering System...")
    print("="*50)
    
    try:
        # Create enhanced interface
        gradio_app = GradioInterface()
        interface = gradio_app.create_enhanced_interface()  # Use enhanced interface
        
        # Launch with enhanced features
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error launching interface: {str(e)}")
