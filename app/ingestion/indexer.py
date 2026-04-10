"""
indexer.py — Document ingestion pipeline: chunk → enrich → embed → persist.

Pipeline:
    1. LOAD    raw docs (loader.py)
    2. CHUNK   TokenTextSplitter + overlap — keeps financial fields together
    3. ENRICH  LlamaIndex LLM-based extractors (title, summary, questions, keywords)
    4. EMBED   OpenAI text-embedding-3-small
    5. PERSIST FAISS index on disk + metadata JSON sidecar
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_openai import OpenAIEmbeddings

from llama_index.core import Document as LlamaDocument
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from app.config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    METADATA_FILE,
    VECTOR_DB_DIR,
)
from app.ingestion.loader import load_documents
from app.ingestion.metadata import extract_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_dump(data: Any, output_path: Path) -> None:
    """Write *data* as pretty-printed JSON, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_doc_metadata(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge file-level fields with rule-based extracted fields.
    Rule-based extraction runs first (cheap, deterministic) so LLM enrichment
    in the pipeline complements rather than replaces it.
    """
    extracted = extract_metadata(
        text=raw_doc["text"],
        category=raw_doc["category"],
        file_name=raw_doc["file_name"],
    )
    return {
        "doc_id":    raw_doc["doc_id"],
        "file_name": raw_doc["file_name"],
        "category":  raw_doc["category"],
        **extracted,
    }


def _to_llama_documents(
    raw_docs: List[Dict[str, Any]],
) -> tuple[List[LlamaDocument], List[Dict[str, Any]]]:
    """
    Convert raw dicts → LlamaIndex Documents with base metadata pre-attached.

    Returns:
        llama_documents  — fed into the ingestion pipeline
        metadata_store   — persisted as JSON sidecar for fast offline filtering
    """
    llama_documents: List[LlamaDocument] = []
    metadata_store: List[Dict[str, Any]] = []

    for raw_doc in raw_docs:
        doc_metadata = _build_doc_metadata(raw_doc)
        metadata_store.append(doc_metadata)
        llama_documents.append(
            LlamaDocument(
                text=raw_doc["text"],
                metadata=doc_metadata,
                id_=str(raw_doc["doc_id"]),
            )
        )

    return llama_documents, metadata_store


def _build_ingestion_pipeline() -> IngestionPipeline:
    """
    LlamaIndex pipeline: split → LLM-enrich metadata.

    Extractor rationale:
    - TitleExtractor        : human-readable headline per chunk
    - SummaryExtractor      : one-line summary aids semantic search
    - QuestionsAnsweredExtractor : surfaces latent Q&A pairs — boosts recall
    - KeywordExtractor      : explicit keywords improve sparse/hybrid search
    """
    llm = LlamaOpenAI(model=CHAT_MODEL, temperature=0.0)

    return IngestionPipeline(
        transformations=[
            TokenTextSplitter(
                separator=" ",
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
            TitleExtractor(nodes=5, llm=llm),
            SummaryExtractor(summaries=["self"], llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
            KeywordExtractor(keywords=8, llm=llm),
        ]
    )


def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce all metadata values to FAISS/pickle-safe scalar types.
    Lists and dicts become JSON strings; None stays None.
    """
    flat: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = str(value)
    return flat


def _nodes_to_langchain_docs(nodes: List[Any]) -> List[LangChainDocument]:
    """
    Convert LlamaIndex nodes → LangChain Documents.

    chunk_id is scoped to (doc_id, position-within-doc) so re-indexing is
    reproducible and IDs don't collide across documents.
    """
    # Track per-doc chunk counters to avoid cross-doc ID collisions
    doc_chunk_counters: Dict[str, int] = {}

    documents: List[LangChainDocument] = []
    for node in nodes:
        try:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
        except Exception:
            text = node.get_content()

        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_id = metadata.get("doc_id", "unknown")

        # Increment per-doc counter → stable, collision-free chunk_id
        doc_chunk_counters[doc_id] = doc_chunk_counters.get(doc_id, 0) + 1
        metadata.setdefault(
            "chunk_id",
            f"{doc_id}_chunk_{doc_chunk_counters[doc_id]:04d}",
        )

        documents.append(
            LangChainDocument(
                page_content=text,
                metadata=_flatten_metadata(metadata),
            )
        )
    return documents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chunks(raw_docs: List[Dict[str, Any]]) -> List[LangChainDocument]:
    """
    Run the full chunk-and-enrich pipeline on *raw_docs*.
    Persists doc-level metadata JSON sidecar as a side-effect.
    """
    llama_documents, metadata_store = _to_llama_documents(raw_docs)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    _safe_json_dump(metadata_store, METADATA_FILE)

    pipeline = _build_ingestion_pipeline()
    nodes = pipeline.run(
        documents=llama_documents,
        in_place=True,
        show_progress=True,
    )

    chunks = _nodes_to_langchain_docs(nodes)
    logger.info("Built %d chunks from %d documents", len(chunks), len(raw_docs))
    return chunks


def build_index(data_dir: Path) -> None:
    """
    Full pipeline entry point: load → chunk → embed → persist FAISS index.
    """
    raw_docs = load_documents(data_dir)
    if not raw_docs:
        raise ValueError(f"No supported documents found in: {data_dir}")

    chunks = build_chunks(raw_docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_DB_DIR))

    logger.info("FAISS index saved to %s (%d chunks)", VECTOR_DB_DIR, len(chunks))
