from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.extractor import DataExtractor

load_dotenv()


class MultiSourceIngestionEngine:

    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.extractor = DataExtractor()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", "•", "-", " "],
        )
        # Local CPU-based embeddings (Free, runs offline, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _process_urls(
        self, urls: List[str], doc_type_override: Optional[str] = None
    ) -> List[Document]:
        docs = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            res = self.extractor.fetch_url(url)
            if res.get("text"):
                inferred_type = doc_type_override or (
                    "culture"
                    if any(
                        k in url.lower()
                        for k in ["about", "culture", "values", "life"]
                    )
                    else "job_description"
                )
                docs.append(
                    Document(
                        page_content=res["text"],
                        metadata={
                            "source_name": url,
                            "source_type": res.get("strategy", "url"),
                            "doc_type": inferred_type,
                            "session_id": self.session_id,
                        },
                    )
                )
        return docs

    def _process_pdfs(
        self, uploaded_files: List[Any], doc_type: str = "job_description"
    ) -> List[Document]:
        docs = []
        for file in uploaded_files:
            try:
                reader = PdfReader(file)
                text_pages = [
                    page.extract_text()
                    for page in reader.pages
                    if page.extract_text()
                ]
                full_text = "\n\n".join(text_pages)

                if full_text.strip():
                    docs.append(
                        Document(
                            page_content=full_text,
                            metadata={
                                "source_name": getattr(
                                    file, "name", "uploaded_doc.pdf"
                                ),
                                "source_type": "pdf_upload",
                                "doc_type": doc_type,
                                "session_id": self.session_id,
                            },
                        )
                    )
            except Exception as e:
                print(
                    f"[Error] Failed to parse PDF"
                    f" {getattr(file, 'name', 'unknown')}: {e}"
                )
        return docs

    def ingest_all(
        self,
        jd_urls: Optional[List[str]] = None,
        culture_urls: Optional[List[str]] = None,
        uploaded_pdfs: Optional[List[Any]] = None,
        raw_text_snippets: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        raw_documents: List[Document] = []

        if jd_urls:
            raw_documents.extend(
                self._process_urls(
                    jd_urls, doc_type_override="job_description"
                )
            )
        if culture_urls:
            raw_documents.extend(
                self._process_urls(culture_urls, doc_type_override="culture")
            )
        if uploaded_pdfs:
            raw_documents.extend(
                self._process_pdfs(uploaded_pdfs, doc_type="job_description")
            )
        if raw_text_snippets:
            for item in raw_text_snippets:
                raw_documents.append(
                    Document(
                        page_content=item["text"],
                        metadata={
                            "source_name": item.get("title", "Pasted Text"),
                            "source_type": "raw_text",
                            "doc_type": item.get(
                                "doc_type", "job_description"
                            ),
                            "session_id": self.session_id,
                        },
                    )
                )

        if not raw_documents:
            raise ValueError(
                "No valid content found across the provided sources."
            )

        chunks = self.splitter.split_documents(raw_documents)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=f"recruiter_{self.session_id}",
        )

        return {
            "vector_db": vector_db,
            "total_documents": len(raw_documents),
            "total_chunks": len(chunks),
            "sources": [d.metadata["source_name"] for d in raw_documents],
        }