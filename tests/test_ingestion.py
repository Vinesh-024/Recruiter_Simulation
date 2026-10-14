import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multi_loader import MultiSourceIngestionEngine

def test_ingestion_pipeline():
    print("Testing Ingestion Engine...")
    engine = MultiSourceIngestionEngine(session_id="test_run_01")
    
    # Test with real URLs (1 culture, 1 job posting)
    test_culture = ["https://stripe.com/about"]
    test_jd_raw = [{
        "title": "Sample AI Engineer JD",
        "doc_type": "job_description",
        "text": """
        Role: AI Application Engineer
        Responsibilities: Build production RAG pipelines, manage vector databases (ChromaDB), 
        orchestrate agents with CrewAI, and develop clean Streamlit interfaces.
        Requirements: Python, LangChain, OpenAI APIs, Docker, Redis, REST APIs.
        """
    }]

    result = engine.ingest_all(
        culture_urls=test_culture,
        raw_text_snippets=test_jd_raw
    )

    print(f"Extraction Successful!")
    print(f"Total Documents: {result['total_documents']}")
    print(f"Total Chunks: {result['total_chunks']}")
    print(f"Indexed Sources: {result['sources']}")

    # Sanity Retrieval Test with metadata filtering
    db = result["vector_db"]
    print("\n--- Testing Metadata-Filtered Retrieval ---")
    
    # 1. Query technical requirements
    tech_chunks = db.similarity_search(
        "vector databases and python", 
        k=1, 
        filter={"doc_type": "job_description"}
    )
    print(f"Technical Match: {tech_chunks[0].page_content[:120]}...")
    
    # 2. Query company culture
    culture_chunks = db.similarity_search(
        "company mission and values", 
        k=1, 
        filter={"doc_type": "culture"}
    )
    print(f"Culture Match: {culture_chunks[0].page_content[:120]}...")

if __name__ == "__main__":
    test_ingestion_pipeline()