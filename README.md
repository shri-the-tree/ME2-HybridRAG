# ME2-HybridRAG 🌊

**Marine Edge Hybrid Retrieval-Augmented Generation System**

A sophisticated hybrid RAG system that combines semantic and keyword search for enhanced document retrieval and question answering, specifically designed for marine industry applications.

## 🚀 Features

### Hybrid Search Engine
- **Semantic Search**: FAISS-powered vector similarity search using HuggingFace embeddings
- **Keyword Search**: BM25Okapi statistical matching for exact term retrieval
- **Smart Score Fusion**: Configurable weighted combination of both search methods
- **Multiple Search Modes**: Runtime switching between semantic, keyword, or hybrid search

### AI Integration
- **Together AI**: Primary LLM provider with DeepSeek-R1-Distill-Llama-70B
- **Gemini Support**: Alternative AI provider configuration
- **Context-Aware Responses**: Enhanced prompts with retrieved document context

### User Interfaces
- **Command Line Interface**: Interactive terminal-based chat
- **Telegram Bot**: Conversational interface for mobile/web access
- **Programmatic API**: Python library for integration into other applications

### Production Features
- **Persistent Storage**: FAISS indices and BM25 models saved to disk
- **Document Processing**: Automated PDF ingestion with intelligent chunking
- **Environment Configuration**: Secure API key and parameter management
- **Error Handling**: Robust exception handling and graceful degradation

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- PDF documents in a designated directory
- API keys for AI services (Together AI and/or Google Gemini)

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shri-the-tree/ME2-HybridRAG.git
   cd ME2-HybridRAG
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   # Required: AI Provider API Keys
   TOGETHER_API_KEY=your_together_ai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional: Telegram Bot (for bot interface)
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   
   # Optional: Custom Directories
   PDF_DIRECTORY=./pdfs
   VECTOR_DB_DIRECTORY=./faiss_db
   ```

4. **Prepare Your Documents**
   
   Place your PDF documents in the `./pdfs` directory (or your configured path)

## 🚀 Quick Start

### Command Line Interface

```bash
python main.py
```

**Available Commands:**
- Regular chat: Type your questions naturally
- `/semantic [query]`: Force semantic-only search
- `/keyword [query]`: Force keyword-only search
- `exit`, `quit`, or `bye`: Exit the application

### Telegram Bot

```bash
python telegram_bot.py
```

Then message your bot on Telegram to start asking questions about your documents.

### Programmatic Usage

```python
from vector_store import HybridMarineEdgeVectorStore
from main import get_bot_response

# Initialize the system
vector_store = HybridMarineEdgeVectorStore(pdf_directory="./pdfs")
vector_store.create_or_load_vector_store()

# Get a response
response = get_bot_response("What are the safety procedures for engine maintenance?")
print(response)
```

## ⚙️ Configuration

### Search Parameters

Edit `config.py` to customize behavior:

```python
# Hybrid search weighting (0.0 = pure keyword, 1.0 = pure semantic)
SEMANTIC_WEIGHT = 0.7

# Default search type
SEARCH_TYPE = "hybrid"  # Options: "hybrid", "semantic", "keyword"

# Document processing
CHUNK_SIZE = 1000        # Text chunk size for embeddings
CHUNK_OVERLAP = 200      # Overlap between chunks
RETRIEVAL_K = 5          # Number of documents to retrieve
```

### Model Selection

```python
# Change the AI model in config.py
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
```

## 🏗️ Architecture

```
ME2-HybridRAG/
├── main.py                 # CLI interface and main orchestration
├── vector_store.py         # Hybrid search engine implementation
├── config.py              # Configuration and environment variables
├── telegram_bot.py        # Telegram bot interface
├── prompt_engineering.py  # System prompts and prompt templates
├── process_pdfs.py        # Document processing utilities
├── requirements.txt       # Python dependencies
├── pdfs/                  # Directory for PDF documents
└── faiss_db/              # Generated vector database storage
```

### Core Components

1. **HybridMarineEdgeVectorStore**: Main search engine combining FAISS and BM25
2. **Together AI Client**: LLM integration for generating responses
3. **Document Processor**: PDF loading and intelligent text chunking
4. **Score Normalizer**: Combines semantic and keyword search scores

## 🔍 How It Works

### Document Ingestion
1. PDFs are loaded from the configured directory
2. Documents are split into overlapping chunks
3. Chunks are embedded using sentence-transformers
4. FAISS index is created for semantic search
5. BM25 index is built for keyword search
6. All indices are persisted to disk

### Query Processing
1. User query is processed by both search engines
2. Semantic search finds conceptually similar content
3. Keyword search finds exact term matches
4. Scores are normalized and combined with configurable weights
5. Top results are formatted as context
6. AI model generates response using retrieved context

### Search Strategies

**Semantic Search** (70% weight by default)
- Best for: Conceptual questions, synonyms, related topics
- Example: "engine problems" → finds "motor issues", "propulsion failures"

**Keyword Search** (30% weight by default)
- Best for: Exact terms, part numbers, specific procedures
- Example: "Model ABC-123" → finds exact model references

**Hybrid Search** (Combined)
- Best for: Most real-world queries requiring both understanding and precision

## 🎯 Use Cases

### Marine Industry Applications
- **Equipment Manuals**: Find maintenance procedures and specifications
- **Safety Protocols**: Retrieve emergency procedures and safety guidelines
- **Regulatory Compliance**: Search through maritime regulations and standards
- **Technical Documentation**: Access engineering specs and technical drawings
- **Incident Reports**: Analyze historical incidents and lessons learned

### Example Queries
```
"What are the maintenance intervals for the main engine?"
"Show me safety procedures for fuel handling"
"Find specifications for Model XYZ-456 pumps"
"What are the requirements for life jacket inspections?"
```

## 🛠️ Development

### Adding New Search Methods

Extend the `HybridMarineEdgeVectorStore` class:

```python
def custom_search(self, query: str, k: int = 10):
    # Implement your search logic
    return results

def query_vector_store(self, query: str, k: int = 5, search_type: str = "hybrid"):
    if search_type == "custom":
        return self.custom_search(query, k)
    # ... existing logic
```

### Custom AI Providers

Add new providers in `config.py` and modify the client initialization in `main.py`.

### Extending Document Types

Modify `process_pdfs.py` to support additional document formats beyond PDF.

## 📊 Performance Tuning

### Memory Usage
- FAISS indices are loaded into memory for fast search
- Typical memory usage: ~100MB per 1000 document chunks
- Use `faiss-gpu` for GPU acceleration with large datasets

### Search Speed
- Semantic search: ~10-50ms per query
- Keyword search: ~1-10ms per query
- Combined hybrid search: ~20-60ms per query

### Accuracy Optimization
- Adjust `SEMANTIC_WEIGHT` based on your use case
- Increase `RETRIEVAL_K` for more context (slower but potentially more accurate)
- Experiment with different embedding models in `vector_store.py`

## 🔧 Troubleshooting

### Common Issues

**"No module named 'faiss'"**
```bash
pip install faiss-cpu
# or for GPU support:
pip install faiss-gpu
```

**NLTK Download Errors**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Empty Search Results**
- Check that PDFs are in the correct directory
- Verify documents contain readable text (not just images)
- Try rebuilding the vector store with `force_reload=True`

**API Errors**
- Verify API keys are correctly set in `.env`
- Check API rate limits and quotas
- Ensure model names are correct and available

## 📄 License

This project is open source. Please refer to the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review the troubleshooting section

## 🙏 Acknowledgments

- **LangChain**: Document processing and text splitting
- **FAISS**: High-performance similarity search
- **Sentence Transformers**: Quality embeddings for semantic search
- **rank-bm25**: BM25 implementation for keyword search
- **Together AI**: LLM inference platform

---

**Made with ❤️ for the Marine Industry**
