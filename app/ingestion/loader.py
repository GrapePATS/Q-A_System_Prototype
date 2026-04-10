"""
loader.py — Document ingestion from data directory.

Loads .txt, .md, and .pdf files from category sub-folders.
Each document is returned as a plain dict so the rest of the pipeline
stays framework-agnostic and easy to test.
"""

import logging
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_file(path: Path) -> str:
    """Dispatch to the correct reader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path) -> List[Dict]:
    """
    Walk *data_dir* and load every supported file.

    Directory layout expected:
        data_dir/
          <category>/
            file1.pdf
            file2.txt

    Returns a list of dicts with keys:
        doc_id  - relative path from data_dir (stable identifier)
        file_name -  bare filename
        category  - name of the immediate parent directory
        text   - extracted plain text (stripped)
    """
    if not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

    docs: List[Dict] = []

    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        for path in sorted(category_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                text = _read_file(path).strip()
            except Exception as exc:
                logger.warning("Skipping %s — could not read: %s", path, exc)
                continue

            if not text:
                logger.warning("Skipping %s — file is empty after extraction.", path)
                continue

            docs.append(
                {
                    "doc_id": str(path.relative_to(data_dir)),
                    "file_name": path.name,
                    "category": category,
                    "text": text,
                }
            )
            logger.debug("Loaded %s (%s)", path.name, category)

    logger.info("Loaded %d documents from %s", len(docs), data_dir)
    return docs