from __future__ import annotations

import streamlit as st

from app.qa.answerer import QAService


st.set_page_config(
    page_title="Thai Securities Research Q&A",
    page_icon="📈",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 3 rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.subtle {
    color: #6b7280;
    font-size: 0.95rem;
}
.metric-card {
    padding: 1rem 1rem 0.75rem 1rem;
    border: 1px solid rgba(49, 51, 63, 0.15);
    border-radius: 14px;
    background: rgba(250, 250, 252, 0.8);
    margin-bottom: 0.75rem;
}
.answer-card {
    padding: 1.2rem 1.2rem 0.8rem 1.2rem;
    border-radius: 16px;
    border: 1px solid rgba(49, 51, 63, 0.15);
    background: #ffffff;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.source-card {
    padding: 0.9rem 1rem 0.7rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(49, 51, 63, 0.12);
    background: #fbfbfd;
    margin-bottom: 0.75rem;
}
.badge {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #eef2ff;
    color: #3730a3;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}
.small-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 0.15rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}
div[data-testid="stExpander"] details {
    border-radius: 12px;
}
</style>
"""

DOC_TYPE_OPTIONS = {
    "ALL": None,
    "_STOCK_RECOMMENDATIONS": "stock_recommendations",
    "MARKET_REPORTS": "market_reports",
    "REGULATIONS": "regulations",
    "COMPANY_PROFILES": "company_profiles",
}

EXAMPLE_QUESTIONS = [
    "What is the current recommendation for PTT stock?",
    "Which stocks have a Buy rating with target price above 100 THB?",
    "What is the P/E ratio of Bangkok Bank?",
    "What are the SEC disclosure requirements?",
]


@st.cache_resource
def get_service() -> QAService:
    return QAService()


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="main-title">📈 Thai Securities Research Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Exploring Thai securities market information with document retrieval</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, str | None, str]:
    with st.sidebar:
        st.markdown("## ⚙️ Search Settings")
        top_k = st.slider("Number of chunks", min_value=3, max_value=12, value=6, step=1)
        selected_label = st.selectbox("Document Type", list(DOC_TYPE_OPTIONS.keys()), index=0)
        doc_type = DOC_TYPE_OPTIONS[selected_label]
        ticker = st.text_input("Ticker filter (optional)", placeholder="e.g., PTT, BBL, AOT").strip().upper()

        st.markdown("---")
        st.markdown("### Example Questions")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"example_{q}"):
                st.session_state["pending_question"] = q

    return top_k, doc_type, ticker


def render_answer(result: dict) -> None:
    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)

    if not sources:
        st.info("No sources to display")
        return

    for src in sources:
        badges = []
        if src.get("category"):
            badges.append(f'<span class="badge">{src["category"]}</span>')
        if src.get("ticker"):
            badges.append(f'<span class="badge">{src["ticker"]}</span>')
        if src.get("rating"):
            badges.append(f'<span class="badge">{src["rating"]}</span>')
        if src.get("report_date"):
            badges.append(f'<span class="badge">{src["report_date"]}</span>')

        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        st.markdown(f"**{src.get('file_name', src.get('doc_id', 'unknown source'))}**")
        if badges:
            st.markdown("".join(badges), unsafe_allow_html=True)

        meta_lines = []
        if src.get("doc_id"):
            meta_lines.append(f"**doc_id:** `{src['doc_id']}`")
        if src.get("analyst"):
            meta_lines.append(f"**analyst:** {src['analyst']}")

        if meta_lines:
            st.markdown(" · ".join(meta_lines))
        st.markdown("</div>", unsafe_allow_html=True)


def render_debug(result: dict) -> None:
    with st.expander("View retrieval / debug details"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
<div class="metric-card">
<div class="small-label">Model</div>
<div><b>{result.get("model", "-")}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
<div class="metric-card">
<div class="small-label">Chunks used</div>
<div><b>{result.get("chunk_count", 0)}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )

        retrieved_chunks = result.get("retrieved_chunks", [])
        if not retrieved_chunks:
            st.warning("No retrieved chunks")
            return

        for i, doc in enumerate(retrieved_chunks, start=1):
            meta = getattr(doc, "metadata", {}) or {}
            with st.expander(f"Chunk {i} — {meta.get('file_name', meta.get('doc_id', 'unknown'))}"):
                st.json(meta)
                st.text(doc.page_content[:2000])


def main() -> None:
    render_header()
    service = get_service()
    top_k, doc_type, ticker = render_sidebar()

    default_question = st.session_state.pop("pending_question", "")
    question = st.text_area(
        "Ask a question",
        value=default_question,
        height=110,
        placeholder="e.g., What is the current recommendation for PTT stock?",
    )

    col1, col2 = st.columns([1, 5])
    ask_clicked = col1.button("🔍 Ask", use_container_width=True)
    clear_clicked = col2.button("Clear", use_container_width=False)

    if clear_clicked:
        st.session_state.pop("last_result", None)
        st.rerun()

    if ask_clicked and question.strip():
        with st.spinner("Searching and summarizing answer..."):
            result = service.ask(
                question=question.strip(),
                top_k=top_k,
                ticker=ticker or None,
                doc_type=doc_type,
            )
            st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            render_answer(result)
            render_debug(result)
        with right:
            render_sources(result.get("sources", []))
    else:
        st.info("Enter a question and click 'Ask' to start searching")


if __name__ == "__main__":
    main()