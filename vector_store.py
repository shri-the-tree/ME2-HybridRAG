import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class HybridMarineEdgeVectorStore:
    def __init__(self, pdf_directory, persist_directory="./faiss_db"):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # FAISS components
        self.faiss_index = None
        self.documents = []

        # BM25 components
        self.bm25 = None
        self.tokenized_docs = []

        # File paths
        self.faiss_index_path = os.path.join(persist_directory, "faiss_index.bin")
        self.documents_path = os.path.join(persist_directory, "documents.pkl")
        self.bm25_path = os.path.join(persist_directory, "bm25.pkl")

    def load_documents(self):
        """Load PDF documents from the specified directory"""
        print(f"Loading documents from {self.pdf_directory}...")
        loader = DirectoryLoader(self.pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks for better retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def preprocess_text(self, text):
        """Tokenize and clean text for BM25"""
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return tokens

    def create_or_load_vector_store(self, force_reload=False):
        """Create a new hybrid vector store or load an existing one"""
        if (os.path.exists(self.faiss_index_path) and
                os.path.exists(self.documents_path) and
                os.path.exists(self.bm25_path) and
                not force_reload):

            print("Loading existing hybrid vector store...")
            self._load_existing_store()
        else:
            print("Creating new hybrid vector store...")
            self._create_new_store()

        return self

    def _load_existing_store(self):
        """Load existing FAISS index, documents, and BM25"""
        # Load FAISS index
        self.faiss_index = faiss.read_index(self.faiss_index_path)

        # Load documents
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)

        # Load BM25
        with open(self.bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data['bm25']
            self.tokenized_docs = bm25_data['tokenized_docs']

    def _create_new_store(self):
        """Create new FAISS index, documents, and BM25"""
        # Load and process documents
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.documents = chunks

        # Create embeddings for FAISS
        print("Creating embeddings...")
        texts = [doc.page_content for doc in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        self.faiss_index.add(embeddings_array)

        # Create BM25 index
        print("Creating BM25 index...")
        self.tokenized_docs = [self.preprocess_text(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Save everything
        self._save_store()

    def _save_store(self):
        """Save FAISS index, documents, and BM25 to disk"""
        os.makedirs(self.persist_directory, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.faiss_index, self.faiss_index_path)

        # Save documents
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        # Save BM25
        bm25_data = {
            'bm25': self.bm25,
            'tokenized_docs': self.tokenized_docs
        }
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(bm25_data, f)

    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Perform semantic search using FAISS"""
        query_embedding = np.array([self.embeddings.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Perform keyword search using BM25"""
        query_tokens = self.preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if idx < len(self.documents) and scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))

        return results

    def hybrid_search(self, query: str, k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[Document, float]]:
        """Perform hybrid search combining semantic and keyword search"""
        # Get results from both methods
        semantic_results = self.semantic_search(query, k=k * 2)
        keyword_results = self.keyword_search(query, k=k * 2)

        # Normalize scores
        semantic_results = self._normalize_scores(semantic_results)
        keyword_results = self._normalize_scores(keyword_results)

        # Combine and weight scores
        combined_scores = {}

        # Add semantic scores
        for doc, score in semantic_results:
            doc_id = id(doc)
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': score,
                'keyword_score': 0.0
            }

        # Add keyword scores
        for doc, score in keyword_results:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = score
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0.0,
                    'keyword_score': score
                }

        # Calculate final weighted scores
        final_results = []
        for doc_info in combined_scores.values():
            final_score = (semantic_weight * doc_info['semantic_score'] +
                           (1 - semantic_weight) * doc_info['keyword_score'])
            final_results.append((doc_info['doc'], final_score))

        # Sort by final score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

    def _normalize_scores(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in results]

        normalized_results = []
        for doc, score in results:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_results.append((doc, normalized_score))

        return normalized_results

    def query_vector_store(self, query: str, k: int = 5, search_type: str = "hybrid") -> List[Tuple[Document, float]]:
        """Query the vector store using specified search type"""
        if not self.faiss_index or not self.bm25:
            self.create_or_load_vector_store()

        if search_type == "semantic":
            return self.semantic_search(query, k)
        elif search_type == "keyword":
            return self.keyword_search(query, k)
        else:  # hybrid
            return self.hybrid_search(query, k)