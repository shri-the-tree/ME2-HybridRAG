from vector_store import HybridMarineEdgeVectorStore
from config import PDF_DIRECTORY, VECTOR_DB_DIRECTORY


def main():
    """Process all PDFs and create/update the hybrid vector store."""
    print("Marine Edge Hybrid PDF Processor")
    print("-" * 50)

    # Initialize and create hybrid vector store
    vector_store = HybridMarineEdgeVectorStore(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=VECTOR_DB_DIRECTORY
    )

    # Force reload the vector store
    vector_store.create_or_load_vector_store(force_reload=True)

    print("Hybrid vector store created and persisted successfully!")
    print(f"Total documents processed: {len(vector_store.documents)}")
    print(f"FAISS index size: {vector_store.faiss_index.ntotal}")
    print(f"BM25 index created with {len(vector_store.tokenized_docs)} documents")


if __name__ == "__main__":
    main()