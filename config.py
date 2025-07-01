# config.py - Configuration settings
import os

class Config:
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LANGUAGE_MODEL = "microsoft/DialoGPT-small"
    
    # Processing settings
    CHUNK_SIZE = 500
    MAX_CHUNKS_FOR_CONTEXT = 3
    MAX_ANSWER_LENGTH = 150
    
    # File paths
    EMBEDDINGS_DIR = "embeddings"
    MODELS_DIR = "models"
    EXPORTS_DIR = "exports"
    
    # Server settings
    HOST = "127.0.0.1"
    PORT = 7860
    DEBUG = True
    
    # Performance settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    TIMEOUT_SECONDS = 300  # 5 minutes
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.EMBEDDINGS_DIR, cls.MODELS_DIR, cls.EXPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)
            
    @classmethod
    def get_model_config(cls):
        """Get model configuration"""
        return {
            'embedding_model': cls.EMBEDDING_MODEL,
            'language_model': cls.LANGUAGE_MODEL,
            'chunk_size': cls.CHUNK_SIZE,
            'max_chunks': cls.MAX_CHUNKS_FOR_CONTEXT,
            'max_length': cls.MAX_ANSWER_LENGTH
        }
