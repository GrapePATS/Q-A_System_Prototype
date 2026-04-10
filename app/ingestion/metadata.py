"""
metadata.py — Rule-based metadata extraction for Thai securities documents.

Extracts structured fields (tickers, ratings, target prices, P/E ratios,
report dates, analyst names) from raw document text so that downstream
retrieval can filter and rank with precision beyond embeddings alone.

Design principle: every field defaults to None / [] so callers can always do
    metadata.get("rating")  without guard checks.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\b([A-Z]{2,6})\b")

# Expanded to cover tokens that appear frequently in these specific documents
_FALSE_TICKER_TOKENS = frozenset({
    "THB", "SET", "SEC", "THE", "AND", "BUY", "HOLD", "SELL",
    "PDF", "EPS", "ROE", "ROA", "NAV", "IPO", "ETF", "GDP",
    "CPI", "YOY", "QOQ", "MOM", "USD", "EUR", "NR",
    # Added: appear heavily in Thai securities docs and are NOT tickers
    "NPL", "NIM", "LNG", "EV", "SME", "BOT", "RSI", "MACD",
    "CAR", "BM", "IPP", "SOTP", "LLM", "AI", "API", "RRF",
})

# Analyst ratings — most specific first to avoid partial matches
_RATING_PATTERNS = [
    (re.compile(r"\bStrong\s+Buy\b", re.I),    "Strong Buy"),
    (re.compile(r"\bStrong\s+Sell\b", re.I),   "Strong Sell"),
    (re.compile(r"\bOutperform\b", re.I),       "Outperform"),
    (re.compile(r"\bUnderperform\b", re.I),     "Underperform"),
    (re.compile(r"\bOverweight\b", re.I),       "Overweight"),
    (re.compile(r"\bUnderweight\b", re.I),      "Underweight"),
    (re.compile(r"\bAccumulate\b", re.I),       "Accumulate"),
    (re.compile(r"\bReduce\b", re.I),           "Reduce"),
    (re.compile(r"\bNeutral\b", re.I),          "Neutral"),
    (re.compile(r"\bBuy\b", re.I),              "Buy"),
    (re.compile(r"\bHold\b", re.I),             "Hold"),
    (re.compile(r"\bSell\b", re.I),             "Sell"),
]

# "target price: 185.00 THB" / "TP 42" / "Target 155"
_TARGET_PRICE_RE = re.compile(
    r"(?:target\s+price|target|TP)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(?:THB|baht|฿)?",
    re.I,
)

# "P/E: 15.2x" / "P/E ratio of 12"
_PE_RE = re.compile(
    r"P/?E(?:\s+ratio)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)(?:\s*[xX])?",
    re.I,
)

# ISO and common short date formats: 2026-03-09 / 9/3/2026 / March 9, 2026
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b"                           # 2026-03-09
    r"|(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"                        # 9/3/2026
    r"|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",                       # March 9, 2026
    re.I,
)

# "Analyst: Napaporn Srisawat" / "Prepared by: Thai Securities Research Team"
_ANALYST_RE = re.compile(
    r"(?:analyst|prepared\s+by)\s*[:\-]\s*([A-Za-z\s]{3,50}?)(?:\n|$)",
    re.I,
)


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

def _extract_tickers(text: str) -> List[str]:
    """Return sorted list of plausible ticker symbols found in the first 5000 chars."""
    candidates = _TICKER_RE.findall(text[:5000])
    return sorted({c for c in candidates if c not in _FALSE_TICKER_TOKENS})


def _extract_rating(text: str) -> Optional[str]:
    """Return the first (most prominent) analyst rating found, or None."""
    for pattern, label in _RATING_PATTERNS:
        if pattern.search(text[:3000]):
            return label
    return None


def _extract_target_price(text: str) -> Optional[float]:
    """Return the first target price as a float, or None."""
    match = _TARGET_PRICE_RE.search(text[:3000])
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_pe_ratio(text: str) -> Optional[float]:
    """Return the first P/E ratio found, or None."""
    match = _PE_RE.search(text[:3000])
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_report_date(text: str) -> Optional[str]:
    """
    Return the first date found in the document as a normalised string, or None.
    Prefers ISO format (YYYY-MM-DD); falls back to whatever matched.
    """
    match = _DATE_RE.search(text[:3000])
    if not match:
        return None
    # Return the first non-None group
    return next((g for g in match.groups() if g), None)


def _extract_analyst(text: str) -> Optional[str]:
    """Return analyst / author name if explicitly labelled, or None."""
    match = _ANALYST_RE.search(text[:3000])
    if match:
        name = match.group(1).strip()
        # Reject implausibly long matches (likely grabbed a whole sentence)
        return name if len(name) <= 60 else None
    return None


def _extract_primary_ticker(tickers: List[str], file_name: str) -> Optional[str]:
    """
    If the filename contains a known ticker (e.g. ptt_research_report.md),
    treat that as the primary ticker; otherwise fall back to the first in list.
    """
    stem = Path(file_name).stem.upper() if file_name else ""
    for ticker in tickers:
        if ticker in stem:
            return ticker
    return tickers[0] if tickers else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata(text: str, category: str, file_name: str) -> Dict[str, Any]:
    """
    Extract structured metadata from a document's raw text.

    All fields are best-effort; missing fields are None or [].

    Args:
        text:      Full document text.
        category:  Directory name used as doc_type (e.g. "stock_recommendations").
        file_name: Bare filename — used for primary ticker heuristic.

    Returns:
        Dict with keys:
            tickers, primary_ticker, rating, target_price,
            pe_ratio, report_date, analyst, doc_type
    """
    tickers = _extract_tickers(text)

    return {
        "tickers":         tickers,
        "primary_ticker":  _extract_primary_ticker(tickers, file_name),
        "rating":          _extract_rating(text),
        "target_price":    _extract_target_price(text),
        "pe_ratio":        _extract_pe_ratio(text),
        "report_date":     _extract_report_date(text),
        "analyst":         _extract_analyst(text),
        "doc_type":        category,
    }