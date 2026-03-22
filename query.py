"""
query.py - Query the RAG system: embed question → search Endee → generate answer
"""

import os
import requests
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
ENDEE_URL   = os.getenv("ENDEE_URL", "http://localhost:8080")
COLLECTION  = "research_papers"
TOP_K       = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "gpt-4o-mini"          # swap with any OpenAI-compatible model

embedder = SentenceTransformer(EMBED_MODEL)
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Endee helpers ─────────────────────────────────────────────────────────
def search_endee(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """Run a nearest-neighbour search against Endee."""
    payload = {
        "vector": query_vector,
        "top_k": top_k,
        "collection": COLLECTION,
    }
    resp = requests.post(f"{ENDEE_URL}/search", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", [])


# ── RAG pipeline ──────────────────────────────────────────────────────────
def answer(question: str) -> dict:
    """
    Full RAG pipeline:
      1. Embed the user question.
      2. Retrieve top-k relevant chunks from Endee.
      3. Build a grounded prompt and call the LLM.
      4. Return answer + sources.
    """
    # Step 1 – embed
    query_vec = embedder.encode(question).tolist()

    # Step 2 – retrieve
    hits = search_endee(query_vec)
    if not hits:
        return {"answer": "No relevant documents found.", "sources": []}

    # Step 3 – build context
    context_blocks = []
    sources = []
    for i, hit in enumerate(hits, 1):
        meta    = hit.get("metadata", {})
        chunk   = meta.get("text", "")
        title   = meta.get("title", "Unknown")
        page    = meta.get("page", "?")
        score   = hit.get("score", 0.0)
        context_blocks.append(f"[{i}] (Source: {title}, p.{page})\n{chunk}")
        sources.append({"rank": i, "title": title, "page": page, "score": round(score, 4)})

    context = "\n\n".join(context_blocks)

    # Step 4 – generate
    system_prompt = (
        "You are a precise research assistant. "
        "Answer the question using ONLY the context provided. "
        "Cite sources by their bracketed numbers, e.g. [1]. "
        "If the context is insufficient, say so clearly."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer_text = completion.choices[0].message.content

    return {"answer": answer_text, "sources": sources}


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    q = " ".join(sys.argv[1:]) or input("Question: ")
    result = answer(q)
    print("\n" + "="*60)
    print("ANSWER\n" + "="*60)
    print(result["answer"])
    print("\nSOURCES")
    for s in result["sources"]:
        print(f"  [{s['rank']}] {s['title']}  p.{s['page']}  (score={s['score']})")
