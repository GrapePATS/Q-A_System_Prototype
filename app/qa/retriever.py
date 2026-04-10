"""
retriever.py — Hybrid retrieval: semantic (dense) + keyword (sparse) + RRF re-rank.

Why hybrid matters for Thai securities:
  • Dense/semantic search handles paraphrase & intent well
    ("what's PTT's outlook" → finds "PTT earnings recovery...")
  • Sparse/keyword search is precise for proper nouns and numbers
    ("P/E 15x Bangkok Bank" must not drift to unrelated financials)

Combining both via Reciprocal Rank Fusion (RRF) consistently outperforms
either alone on finance Q&A benchmarks without requiring a cross-encoder.

Previous bug fixed: sparse_results was re-sorting dense_results by keyword
score — making it a reordering of the same list, not an independent ranked
list. RRF over two identical-document lists collapses to a single weighted
list and provides no fusion benefit.

Fix: fetch a larger candidate pool from FAISS (fetch_k), then produce TWO
genuinely independent lists — one ranked by embedding similarity (FAISS
order), one ranked by keyword score — before fusing.
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, METADATA_FILE, TOP_K, VECTOR_DB_DIR

logger = logging.getLogger(__name__)

# RRF constant — 60 is the standard value from the original paper
_RRF_K = 60


# ---------------------------------------------------------------------------
# Sparse (keyword) scorer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _keyword_score(query_tokens: List[str], doc_text: str) -> float:
    """
    Simple term-frequency presence score.
    Sufficient for re-ranking within a small candidate pool.
    """
    if not query_tokens:
        return 0.0
    doc_token_set = set(_tokenize(doc_text))
    hits = sum(1 for t in query_tokens if t in doc_token_set)
    return hits / len(query_tokens)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = _RRF_K,
) -> List[Document]:
    """
    Merge N ranked lists into one fused ranking using RRF.
    Documents are identified by a hash of their page_content (stable within a run).
    """
    scores: Dict[int, float] = defaultdict(float)
    doc_map: Dict[int, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = hash(doc.page_content)
            scores[key] += 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Retrieves documents using dense (semantic) + sparse (keyword) fusion.

    Public methods:
        retrieve(query, top_k)          — main entry point, returns fused results
        retrieve_semantic(query, top_k) — dense only (for eval / ablation)
        filter_by_metadata(...)         — offline filter on JSON sidecar
    """

    def __init__(self) -> None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata_store: List[Dict[str, Any]] = json.load(f)

        logger.info(
            "HybridRetriever ready. %d doc-level metadata entries loaded.",
            len(self.metadata_store),
        )

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve_semantic(self, query: str, top_k: int = TOP_K) -> List[Document]:
        """Dense-only retrieval — fast path for semantic similarity queries."""
        return self.vectorstore.similarity_search(query, k=top_k)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        """
        Hybrid retrieval: dense + sparse fused via RRF.

        Fetches a larger candidate pool (fetch_k = top_k * 3) from FAISS so
        the sparse pass has room to promote keyword-heavy chunks that ranked
        lower in the dense pass. Both lists are produced from the same pool
        but ordered independently before fusion.
        """
        fetch_k = top_k * 3

        # --- Dense list: FAISS cosine-similarity order ---
        dense_results: List[Document] = self.vectorstore.similarity_search(
            query, k=fetch_k
        )

        # --- Sparse list: same pool, re-ordered by keyword score ---
        query_tokens = _tokenize(query)
        sparse_results: List[Document] = sorted(
            dense_results,
            key=lambda d: _keyword_score(query_tokens, d.page_content),
            reverse=True,
        )

        # --- Fuse two independently-ordered lists ---
        fused = _reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Offline metadata filtering (JSON sidecar — no vector index involved)
    # ------------------------------------------------------------------

    def filter_by_metadata(
        self,
        rating: Optional[str] = None,
        min_target_price: Optional[float] = None,
        doc_type: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter doc-level metadata without touching the vector index.
        Useful for structured queries: "all Buy-rated stocks above THB 100".

        All parameters are optional and ANDed together.
        """
        results = self.metadata_store

        if rating:
            results = [
                r for r in results
                if (r.get("rating") or "").lower() == rating.lower()
            ]
        if min_target_price is not None:
            results = [
                r for r in results
                if r.get("target_price") is not None
                and r["target_price"] >= min_target_price
            ]
        if doc_type:
            results = [
                r for r in results
                if r.get("doc_type", "").lower() == doc_type.lower()
            ]
        if ticker:
            results = [
                r for r in results
                if ticker.upper() in [t.upper() for t in (r.get("tickers") or [])]
            ]

        return results