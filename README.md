# Thai Securities Market Intelligence — RAG Q&A Prototype

## Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Approach](#2-architecture--approach)
3. [Setup & Run Guide](#3-setup--run-guide)
4. [Project Structure](#4-project-structure)
5. [Assumptions](#5-assumptions)
6. [Known Limitations](#6-known-limitations)
7. [Ideas for Future Improvements](#7-ideas-for-future-improvements)

---

## 1. Project Overview

An intelligent Q&A prototype that allows users to ask natural-language questions about Thai securities market information and receive accurate, grounded answers with source attribution.

The system uses a Retrieval-Augmented Generation (RAG) pipeline built on top of:

| Component | Role |
|---|---|
| **LlamaIndex** | Document chunking and LLM-based metadata enrichment |
| **FAISS** | Local vector store for fast similarity search |
| **OpenAI** | Embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`) |
| **LangChain** | Orchestration glue between retrieval and generation |

**Supported document types:** `.md` · `.txt` · `.pdf`

**Supported document categories:**

- Daily market reports
- Stock research reports
- SEC / SET regulations
- Company profiles

---

## 2. Architecture & Approach

The system is organised into three clearly separated layers:

```
┌─────────────────────────────────────────────────────────┐
│  INGESTION LAYER  (run once, or when documents change)  │
│                                                         │
│  loader.py     → reads .md / .txt / .pdf by category   │
│  metadata.py   → rule-based field extraction:           │
│                  tickers, rating, target price, P/E,    │
│                  report date, analyst name              │
│  indexer.py    → LlamaIndex pipeline:                   │
│                  TokenTextSplitter (800 tok, 120 ovlp)  │
│                  + TitleExtractor                        │
│                  + SummaryExtractor                      │
│                  + QuestionsAnsweredExtractor            │
│                  + KeywordExtractor                      │
│                  → FAISS index + metadata_store.json    │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                         │
│                                                         │
│  retriever.py  → HybridRetriever:                       │
│                  1. Dense pass  — FAISS cosine search   │
│                  2. Sparse pass — keyword TF scoring    │
│                  3. Fuse via Reciprocal Rank Fusion      │
│                  filter_by_metadata() — offline filter   │
│                  on JSON sidecar (ticker, rating, etc.) │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  ANSWER LAYER                                            │
│                                                         │
│  answerer.py   → QAService:                             │
│                  optional metadata pre-filter           │
│                  → retrieve hybrid chunks               │
│                  → format context with enriched fields  │
│                  → call gpt-4o-mini with system prompt  │
│                  → return {answer, sources, chunks,     │
│                            chunk_count, model}          │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**a) Hybrid retrieval (dense + sparse + RRF)**
Pure embedding search drifts on proper nouns and numbers — "P/E 7.3x Bangkok Bank" may retrieve unrelated financials. A lightweight keyword scorer runs over the same candidate pool and the two rankings are fused via Reciprocal Rank Fusion (k=60). This consistently outperforms either method alone on finance Q&A without requiring a heavyweight cross-encoder.

**b) Two-pass metadata enrichment**
Rule-based extraction (`metadata.py`) runs first — deterministic and cheap — producing tickers, rating, target price, date, and analyst name directly from the text. LlamaIndex LLM extractors then add title, one-line summary, latent Q&A pairs, and keywords per chunk. The LLM pass complements rather than replaces the rule-based pass, ensuring critical financial fields are always present even if the LLM extractor is slow or fails.

**c) JSON metadata sidecar**
Doc-level metadata is persisted separately from the FAISS index as `metadata_store.json`. This enables zero-cost structured filtering (e.g. "all BUY-rated stocks with target > THB 100") without running a vector search at all, and supports offline analytics.

**d) Grounded system prompt**
The LLM is instructed to answer exclusively from provided context using , use a specific fallback phrase when information is absent, and always conclude with a Sources section citing folder/filename. This keeps hallucination risk low and makes attribution auditable.

**e) Chunk design**
Chunk size of 800 tokens with 120-token overlap. The overlap ensures financial tables are not split in a way that loses column headers. The 800-token limit keeps each chunk semantically focused for embedding quality.

---

## 3. Setup & Run Guide

### Requirements
- Python 3.10 or higher
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`

### Step 1 — Clone the repository

```bash
git clone https://github.com/GrapePATS/Q-A_System_Prototype.git
cd Q-A_System_Prototype
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Key packages: `langchain` `langchain-openai` `langchain-community` `faiss-cpu` `llama-index-core` `llama-index-llms-openai` `llama-index-embeddings-openai` `pypdf` `python-dotenv`

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=sk-...                        # Required
EMBEDDING_MODEL=text-embedding-3-small       # Optional — this is the default
CHAT_MODEL=gpt-4o-mini                       # Optional — this is the default
```

#### How to Get an OpenAI API Key
1. Sign Up --> Go to https://platform.openai.com/ and create an account.
2. Create API Key
- Visit: https://platform.openai.com/api-keys
- Click **Create new secret key**
- Copy and save it
3. Add to `.env`

### Step 5 — Run the application
```
streamlit run main.py
```
Then open the local URL shown in your terminal (usually http://localhost:8501).

## 4. Project Structure

```
app/
  config.py                  — paths, model names, chunk parameters
  ingestion/
    loader.py                — reads .txt / .md / .pdf by category folder
    metadata.py              — rule-based field extraction (ticker, rating, etc.)
    indexer.py               — chunk → LLM-enrich → embed → persist
  qa/
    retriever.py             — HybridRetriever (dense + sparse + RRF)
    answerer.py              — QAService (retrieval → LLM → structured response)
    prompts.py               — system prompt for the Thai Market Intelligence role

data/                        — place source documents here
storage/
  faiss_index/               — generated by build_index (git-ignored)
  metadata_store.json        — generated by build_index (git-ignored)

.env                         — API keys (never commit)
.env.example                 — template for environment variables
requirements.txt             — Python dependencies
```

---

## 5. Assumptions
- Pipeline Efficacy: The overall system performance is fundamentally driven by the synergy between strategic Chunking, robust Hybrid Retrieval, and Grounded Generation.
- Metadata Centrality: High-fidelity metadata is the primary catalyst for improving retrieval precision and maintaining context within individual data chunks.

---

## 6. Limitations

- Structural Integrity of Tables: Since TokenTextSplitter operates on token count rather than semantic structure, extensive financial tables (exceeding 800 tokens) risk being bifurcated. While a 120-token overlap mitigates this, it does not fully guarantee the preservation of table headers and relational context.
- Heuristic Metadata Extraction: The current regex-based approach is effective for standardized patterns but lacks the flexibility to capture non-conventional formats, such as Thai numerals or vernacular rating descriptions.
- Ingestion Overhead & Scalability: Running four LLM-based extractors per chunk incurs significant latency and API costs during ingestion. Furthermore, the lack of an incremental update mechanism necessitates a full re-indexing of the corpus for any change.
- Cross-Chunk Synthesis: The system currently faces challenges in synthesizing information that is fragmented across disparate chunks or multiple document paths, occasionally leading to incomplete retrieval for complex queries.
- Recommendation Nuance: The quality of stock recommendations remains in the prototype stage; further Prompt Tuning is required to achieve professional-grade financial advisory tone and depth.

---

## 7. Ideas for Future Improvements

- Advanced Document Classification: Implement an automated doc_type classifier to improve retrieval routing and metadata application.
- Enhanced Metadata Architecture: Upgrade from simple rule-based extraction to an LLM-powered or Hybrid-Entity Recognition (NER) model to capture complex financial entities more accurately.
- Tabular Data Understanding: Integrate Table-aware chunking strategies or convert complex tables into simpler structured  representations to maintain data integrity across splits.
- Incremental Indexing Logic: Develop a content-hashing mechanism to identify and process only new or modified documents, significantly reducing ingestion time and operational costs.
- Multi-hop Reasoning: Explore Agentic RAG or Recursive Retrieval patterns to better connect information scattered across multiple documents or sectors.
