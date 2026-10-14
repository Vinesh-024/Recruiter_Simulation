Recruiter Simulator: Multi-Agent RAG Interview PlatformAn end-to-end AI interview simulation platform that roleplays a Senior Hiring Manager and an Executive Career Coach. The system dynamically ingests live job postings, ATS feeds, company culture pages, and candidate resumes to run realistic, role-tailored technical and behavioral mock interviews with real-time STAR scoring.Architecture OverviewPlaintext ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                           INGESTION & EXTRACTION LAYER                          │
 │                                                                                 │
 │   [ ATS URLs ]              [ Web URLs ]             [ Campus PDFs / Resumes ] │
 │   Greenhouse / Lever API    Jina Reader Proxy        PyPDF In-Memory Parser    │
 └─────────────┬─────────────────────┬──────────────────────────────┬──────────────┘
               │                     │                              │
               ▼                     ▼                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                        UNIFIED NORMALIZER & CHUNKER                             │
 │                                                                                 │
 │   • RecursiveCharacterTextSplitter (Chunk: 450 tokens, Overlap: 50 tokens)      │
 │   • Metadata Enrichment: { doc_type: culture | job_description, source, role }  │
 └────────────────────────────────────┬────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                        SESSION CHROMADB VECTOR STORE                            │
 │                                                                                 │
 │   • Embeddings: Local sentence-transformers/all-MiniLM-L6-v2                    │
 │   • Ephemeral Collection Isolation: `recruiter_{session_id}`                    │
 └────────────────────┬─────────────────────────────────────────────┬──────────────┘
                      │                                             │
      Filter: doc_type="culture"                    Filter: doc_type="job_description"
                      │                                             │
                      ▼                                             ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                            CREWAI MULTI-AGENT LOOP                              │
 │                                                                                 │
 │   ┌───────────────────────────┐             ┌───────────────────────────────┐   │
 │   │     INTERVIEWER AGENT     │             │     CANDIDATE COACH AGENT     │   │
 │   │ (Senior Hiring Manager)   │             │   (STAR & Keyword Evaluator)  │   │
 │   └─────────────┬─────────────┘             └───────────────▲───────────────┘   │
 │                 │                                           │                   │
 │                 │ 1. Grounded Question                      │ 3. Pydantic-typed │
 │                 │    (Tech / Behavioral)                    │    STAR Scorecard │
 │                 ▼                                           │                   │
 │          [ CANDIDATE (YOU) ] ── 2. Raw STAR Response ───────┘                   │
 │                 ▲                                                               │
 │                 │ 4. Adaptive Follow-up on Distilled Weak Spot                  │
 │                 └───────────────────────────────────────────────────────────────┘
Key Features & Engineering Highlights1. 3-Layer Graceful Degradation IngestionUnlike standard RAG demos that rely on pre-cleaned local text files, this pipeline dynamically extracts unstructured company data at runtime:Layer 1 (Direct ATS APIs): Intercepts public JSON endpoints on Greenhouse (boards-api.greenhouse.io) and Lever (api.lever.co) for 100% accurate, zero-scraping job data.Layer 2 (Jina Reader API): Proxies company /about and custom career pages through [https://r.jina.ai/](https://r.jina.ai/) to bypass client-side JavaScript hydration and strip cookie banners/navbars into clean Markdown.Layer 3 (Trafilatura Fallback): Runs an in-memory, zero-dependency local scraper to extract main article text if external endpoints time out.Multi-Modal Campus Mode: Parses single or multi-role campus recruitment PDFs via pypdf, preserving bulleted qualification requirements.2. Context Slicing & Metadata IsolationAvoids vector space contamination by tagging all chunks with doc_type: "culture" vs doc_type: "job_description".Behavioral questions trigger vector searches strictly against culture documents.Technical questions query hard-skill requirements, preventing company values from diluting technical prompts.3. Agent State Distillation (Token Economy)The Coach evaluates answers against the STAR framework and outputs a validated Pydantic v2 schema.Rather than feeding full coaching critiques into the Interviewer's prompt (which causes prompt bloat and role bleed), the system extracts a single distilled directive string (probe_target) for the next question.4. Resume-Grounded CoachingIngests the candidate's actual resume into in-memory session context.When the Coach recommends an improved answer rewrite, it references the user's real project history and tools rather than hallucinating generic examples.Tech StackComponentTechnologyRationaleAgent OrchestrationCrewAIRole-based agent personas, custom tool bindings, and task execution.Inference EngineOpenAI gpt-4o-miniLow-latency, cost-effective reasoning for conversational turns.Embeddingstext-embedding-3-small1536-dimensional semantic representations with high retrieval efficiency.Vector DatabaseChromaDBIn-process, zero-configuration vector store with session-isolated collections.RAG Glue & ChunkingLangChain Core / Text SplittersStructural boundary splitting (\n## , \n### , •).Web Ingestionhttpx, BeautifulSoup4, trafilaturaResilient 3-layer web scraping cascade.PDF ProcessingpypdfFast extraction of campus brochures and resume documents.Schema ValidationPydantic v2Enforces type safety on structured model outputs.Frontend UIStreamlitReal-time dual-column chat and evaluation dashboard.Project StructurePlaintextrecruiter-simulator/
├── chromadb_store/          # Ephemeral local vector storage
├── src/
│   ├── __init__.py          # Package exports
│   ├── extractor.py         # 3-Layer Extraction Cascade (ATS -> Jina -> Trafilatura)
│   ├── multi_loader.py      # Unified Normalizer, Chunker, & ChromaDB Ingestion
│   ├── retriever.py         # Custom metadata-filtered CrewAI tools
│   ├── agents.py            # Interviewer & Coach agents with Pydantic schemas
│   └── app.py               # Streamlit dual-column dashboard
├── tests/
│   └── test_ingestion.py    # Pipeline sanity check script
├── .env.example             # Environment variable template
├── requirements.txt         # Project dependencies
└── README.md
Quickstart Guide1. Clone & Setup EnvironmentBashgit clone https://github.com/yourusername/recruiter-simulator.git
cd recruiter-simulator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Configure Environment VariablesCreate a .env file in the root directory:Bashcp .env.example .env
Add your Groq API key to `.env` as `GROQ_API_KEY=your-key`. Embeddings run locally and do not require an OpenAI key.
3. Run the offline regression tests: `python -m unittest discover -s tests -p "test_*.py"`
4. Launch the application: `streamlit run src/app.py`

Paste a job description to start a role-specific practice session. Company culture URLs and a PDF resume are optional. Sessions are held in memory and removed when you start a new interview or delete current practice data.
Portfolio & Interview Talking PointsBe prepared to discuss these design choices in technical interviews:1. "Why not just use a standard 'Chat with PDF' RAG pipeline?""Standard RAG demos dump an entire PDF into fixed-size chunks. In job simulations, this fails because job descriptions contain dense bulleted requirements, while company culture pages contain long-form narratives. Mixing them dilutes vector relevance. I designed a metadata-filtered ingestion engine that tags chunks by doc_type and dynamically routes queries based on whether the question is technical or behavioral."2. "How did you solve HTML noise and JS hydration during web extraction?""Raw scraping often injects cookie banners, navbars, and unrendered React components into the vector store, polluting semantic search. I implemented a 3-layer extraction cascade: direct public ATS APIs (Greenhouse/Lever) first for structured JSON, Jina Reader API as a secondary proxy to convert arbitrary career pages into clean Markdown, and local Trafilatura as an offline fallback."3. "How do you prevent the agents from getting confused or hallucinating?""I implemented Context Slicing and Inter-Agent State Distillation. The Interviewer Agent never sees the full coaching rubric or the candidate's full resume—it only receives the retrieved JD context and a distilled 1-line probe_target directive from the Coach. The Coach outputs a validated Pydantic v2 schema, ensuring structured metrics in the UI and deterministic handoffs between agents."4. "How do you manage token economics across multi-turn interviews?""Appending full conversation transcripts causes quadratic token growth. I utilized a sliding window of 2 turns for active agent memory, capped vector retrieval at $k=2$ chunks using semantic chunking (~450 tokens), and offloaded session history rendering entirely to the Streamlit UI state."LicenseMIT License. Free for educational, portfolio, and personal use.