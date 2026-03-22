"""
ui.py - Streamlit frontend for the Endee RAG system
Run: streamlit run ui.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ResearchRAG · Endee",
    page_icon="🔬",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; margin: auto; }
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #7c3aed;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .answer-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        font-size: 0.95rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.title("🔬 ResearchRAG")
st.caption("Ask questions about your research papers — powered by **Endee** vector database")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g. What is the attention mechanism in transformers?",
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_btn = st.button("Ask", type="primary", use_container_width=True)
with col2:
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)

# ── Query ─────────────────────────────────────────────────────────────────
if ask_btn and question.strip():
    with st.spinner("Searching Endee and generating answer …"):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            st.subheader("Answer")
            st.markdown(
                f'<div class="answer-box">{data["answer"]}</div>',
                unsafe_allow_html=True,
            )

            if data.get("sources"):
                st.subheader(f"Retrieved Sources ({len(data['sources'])})")
                for s in data["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>[{s["rank"]}]</b> {s["title"]} &nbsp;·&nbsp; '
                        f'Page {s["page"]} &nbsp;·&nbsp; '
                        f'Score: {s["score"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure `app.py` is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

elif ask_btn:
    st.warning("Please enter a question first.")

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with [Endee](https://github.com/endee-io/endee) · sentence-transformers · OpenAI · FastAPI")
