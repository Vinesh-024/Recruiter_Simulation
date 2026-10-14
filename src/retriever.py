from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from crewai.tools import tool


def _format_retrieved_chunks(chunks: List[Document]) -> str:
    """Formats retrieved LangChain Document chunks into structured markdown for LLMs."""
    if not chunks:
        return "No relevant context found in the knowledge base."

    formatted = []
    for i, doc in enumerate(chunks, 1):
        source = doc.metadata.get("source_name", "Unknown Source")
        doc_type = doc.metadata.get("doc_type", "general")
        formatted.append(
            f"--- Context Chunk {i} [{doc_type.upper()} | Source: {source}] ---\n"
            f"{doc.page_content.strip()}\n"
        )
    return "\n".join(formatted)


def get_recruiter_tools(vector_db: Chroma):
    """
    Factory function returning instantiated CrewAI tools bound to a specific
    session ChromaDB vector store.
    """

    @tool("query_job_requirements")
    def query_job_requirements(query: str) -> str:
        """
        Search specifically for job description requirements, hard skills,
        tech stacks, responsibilities, and qualifications.
        Use this tool when generating technical interview questions or
        checking for required candidate skills.
        """
        try:
            results = vector_db.similarity_search(
                query=query,
                k=2,
                filter={"doc_type": "job_description"}
            )
            return _format_retrieved_chunks(results)
        except Exception as e:
            return f"Error retrieving job requirements: {str(e)}"

    @tool("query_company_culture")
    def query_company_culture(query: str) -> str:
        """
        Search specifically for company values, leadership principles,
        engineering culture, and behavioral expectations.
        Use this tool when formulating behavioral questions or evaluating
        how well a candidate aligns with the company's operating principles.
        """
        try:
            results = vector_db.similarity_search(
                query=query,
                k=2,
                filter={"doc_type": "culture"}
            )
            return _format_retrieved_chunks(results)
        except Exception as e:
            return f"Error retrieving company culture: {str(e)}"

    @tool("query_unified_context")
    def query_unified_context(query: str) -> str:
        """
        Search across all ingested documents (job descriptions, culture pages,
        and placement PDFs) without metadata filtering.
        Use this tool for broad queries spanning both technical and organizational topics.
        """
        try:
            results = vector_db.similarity_search(query=query, k=3)
            return _format_retrieved_chunks(results)
        except Exception as e:
            return f"Error searching context: {str(e)}"

    return [query_job_requirements, query_company_culture, query_unified_context]