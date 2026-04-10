"""
answerer.py — QAService: orchestrates retrieval → prompt → LLM → structured response.

Responsibility boundary:
  - QAService does NOT know about Streamlit, CLI, or any UI layer.
  - It always returns a plain dict so any frontend can consume it.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, TOP_K
from app.qa.prompts import QA_SYSTEM_PROMPT
from app.qa.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class QAService:
    """
    Orchestrates the full RAG pipeline for a single question.

    Usage:
        service = QAService()
        result  = service.ask("What is PTT's target price?")

        result["answer"]            — LLM-generated answer string
        result["sources"]           — deduplicated source metadata list
        result["retrieved_chunks"]  — raw Document list for eval / debug
        result["chunk_count"]       — number of chunks used (for eval)
        result["model"]             — model name used (for eval)
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        """
        Render retrieved chunks as numbered source blocks for the prompt.
        Includes LlamaIndex-enriched fields (summary, keywords) and
        securities-specific fields (ticker, rating, report_date) when present,
        so the LLM has rich context without needing to infer them from raw text.
        """
        parts = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata

            # Core attribution fields
            header_lines = [
                f"[Source {i}]",
                f"Document : {meta.get('file_name', meta.get('doc_id', 'unknown'))}",
                f"Category : {meta.get('category', 'unknown')}",
            ]

            # Securities-specific fields — only added when present
            for label, key in [
                ("Ticker",      "primary_ticker"),
                ("Rating",      "rating"),
                ("Date",        "report_date"),
                ("Analyst",     "analyst"),
                ("Summary",     "section_summary"),
                ("Keywords",    "excerpt_keywords"),
            ]:
                value = meta.get(key)
                if value:
                    header_lines.append(f"{label:<10}: {value}")

            parts.append("\n".join(header_lines) + f"\n\nContent:\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _deduplicate_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Return one entry per unique doc_id preserving first-seen order.
        Includes securities-specific fields for rich frontend attribution.
        None values are stripped so the response stays clean.
        """
        seen: set = set()
        sources: List[Dict[str, Any]] = []

        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if doc_id in seen:
                continue
            seen.add(doc_id)

            meta = doc.metadata
            entry = {
                "doc_id":      doc_id,
                "category":    meta.get("category"),
                "file_name":   meta.get("file_name"),
                "ticker":      meta.get("primary_ticker"),
                "rating":      meta.get("rating"),
                "report_date": meta.get("report_date"),
                "analyst":     meta.get("analyst"),
            }
            sources.append({k: v for k, v in entry.items() if v is not None})

        return sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = TOP_K,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer *question* using hybrid retrieval + LLM generation.

        Args:
            question: Natural language question from the user.
            top_k:    Number of chunks to retrieve (default from config).
            ticker:   Optional ticker filter applied before retrieval
                      e.g. "BBL" restricts search to BBL documents only.
            doc_type: Optional document type filter
                      e.g. "stock_recommendations" | "market_reports" |
                           "regulations" | "company_profiles"

        Returns:
            {
                "answer":           str   — LLM-generated answer,
                "sources":          list  — deduplicated source metadata,
                "retrieved_chunks": list  — raw Document objects for eval/debug,
                "chunk_count":      int   — number of chunks used,
                "model":            str   — LLM model name,
            }
        """
        # ------------------------------------------------------------------
        # Optional metadata pre-filter: restrict to matching doc_ids before
        # hitting the vector index — prevents cross-stock confusion on
        # ticker-specific questions (e.g. "BBL's NPL" retrieving KBANK chunks)
        # ------------------------------------------------------------------
        allowed_ids: Optional[set] = None
        if ticker or doc_type:
            filtered_meta = self.retriever.filter_by_metadata(
                ticker=ticker,
                doc_type=doc_type,
            )
            allowed_ids = {m["doc_id"] for m in filtered_meta}
            logger.debug(
                "Metadata pre-filter (ticker=%s, doc_type=%s): %d docs allowed",
                ticker, doc_type, len(allowed_ids),
            )

        # ------------------------------------------------------------------
        # Retrieve
        # ------------------------------------------------------------------
        docs = self.retriever.retrieve(question, top_k=top_k)

        if allowed_ids is not None:
            docs = [d for d in docs if d.metadata.get("doc_id") in allowed_ids]
            logger.debug("After metadata post-filter: %d chunks remain", len(docs))

        # ------------------------------------------------------------------
        # Fallback: nothing retrieved — return early without calling LLM
        # ------------------------------------------------------------------
        if not docs:
            logger.warning(
                "No chunks retrieved for question: %s (ticker=%s, doc_type=%s)",
                question, ticker, doc_type,
            )
            return {
                "answer":           "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มี กรุณาระบุคำถามให้ชัดเจนขึ้น หรือตรวจสอบว่าเอกสารที่ต้องการถูก index แล้ว",
                "sources":          [],
                "retrieved_chunks": [],
                "chunk_count":      0,
                "model":            CHAT_MODEL,
            }

        # ------------------------------------------------------------------
        # Build prompt and call LLM
        # ------------------------------------------------------------------
        context = self._format_context(docs)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Provide a concise answer based only on the context above, "
            "and list the [Source N] references used."
        )

        response = self.llm.invoke(
            [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]
        )

        return {
            "answer":           response.content,
            "sources":          self._deduplicate_sources(docs),
            "retrieved_chunks": docs,
            "chunk_count":      len(docs),
            "model":            CHAT_MODEL,
        }
