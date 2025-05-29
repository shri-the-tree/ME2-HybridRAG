import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"


# Bot configuration
MAX_HISTORY_LENGTH = 10  # Number of messages to keep in history

# Vector database configuration
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")  # Directory containing your PDFs
VECTOR_DB_DIRECTORY = os.getenv("VECTOR_DB_DIRECTORY", "./faiss_db")  # Directory to store vector database
CHUNK_SIZE = 1000  # Size of text chunks for embedding
CHUNK_OVERLAP = 200  # Overlap between chunks
RETRIEVAL_K = 5  # Number of relevant chunks to retrieve

# Hybrid search configuration
SEMANTIC_WEIGHT = 0.7  # Weight for semantic search (0.0-1.0, higher = more semantic)
SEARCH_TYPE = "hybrid"  # Options: "hybrid", "semantic", "keyword"