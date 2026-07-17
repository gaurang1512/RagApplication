from src.data_loader import load_all_documents 
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


#Example usage


if __name__ == "__main__":
    # Initialize RAG Search
    rag = RAGSearch(persist_dir="faiss_store")
    
    # Example query
    query = "What are the Types of SECURITY?"
    print(f"\n[USER QUERY] {query}\n")
    
    # Get summarized response
    summary = rag.search_and_summarize(query, top_k=3)
    print(f"[SUMMARY]\n{summary}\n")
    