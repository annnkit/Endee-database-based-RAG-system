# 🔬 ResearchRAG — Retrieval-Augmented Generation with Endee

> Ask natural language questions over a corpus of research papers.  
> Powered by **[Endee](https://github.com/endee-io/endee)** — a high-performance vector database handling up to 1B vectors on a single node.

---

## 📌 Problem Statement

Academic researchers and engineers often need to extract insights from large collections of papers,
reports, or technical documents. Traditional keyword search fails to capture *meaning* — searching
for "neural attention" won't return documents that only mention "transformer self-attention weights."

**ResearchRAG** solves this by:
1. Converting every document chunk into a semantic vector using a sentence embedding model.
2. Storing all vectors in **Endee** for ultra-fast cosine similarity search.
3. Retrieving the most semantically relevant chunks for any question.
4. Feeding those chunks to an LLM to generate a grounded, cited answer.

---

## 🏗️ System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  PDF / TXT  ──►  Chunker  ──►  SentenceTransformer  ──►  Endee  │
│  (data/)         (400 words     (all-MiniLM-L6-v2)    (upsert)  │
│                   overlap=50)    384-dim vectors                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                  │
│  User Question                                                   │
│       │                                                          │
│       ▼                                                          │
│  SentenceTransformer  ──►  Endee  ──►  Top-K Chunks             │
│  (embed question)          (ANN          (with metadata)        │
│                             search)           │                  │
│                                               ▼                  │
│                                          LLM (GPT-4o-mini)      │
│                                          "Answer + Citations"    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         API + UI LAYER                           │
│                                                                  │
│  Streamlit UI  ──►  FastAPI (/ask)  ──►  query.py               │
└──────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Role |
|-----------|------|
| **Endee** | Vector database — stores 384-dim embeddings, serves ANN queries via HTTP |
| **sentence-transformers** | Converts text → dense vectors (all-MiniLM-L6-v2) |
| **ingest.py** | Reads docs, chunks text, embeds chunks, upserts to Endee |
| **query.py** | Embeds question, queries Endee, calls LLM with retrieved context |
| **app.py** | FastAPI REST API exposing `/ask` endpoint |
| **ui.py** | Streamlit web UI for interactive Q&A |

---

## 🔑 How Endee Is Used

Endee is the **core retrieval engine** of this project. It replaces heavier alternatives like
Pinecone, Weaviate, or Qdrant with a self-hosted, high-performance option.

### 1. Collection creation
```python
requests.post("http://localhost:8080/collections", json={
    "name": "research_papers",
    "dimension": 384,          # matches all-MiniLM-L6-v2 output size
    "distance": "cosine",
})
```

### 2. Upserting vectors (ingestion)
```python
requests.post("http://localhost:8080/upsert", json={
    "collection": "research_papers",
    "vectors": [
        {
            "id": "<uuid>",
            "vector": [0.12, -0.45, ...],   # 384 floats
            "metadata": {
                "text":   "chunk content …",
                "title":  "attention_is_all_you_need",
                "page":   3,
                "source": "data/attention_is_all_you_need.pdf"
            }
        }
    ]
})
```

### 3. Similarity search (query time)
```python
requests.post("http://localhost:8080/search", json={
    "collection": "research_papers",
    "vector": [0.08, -0.31, ...],   # embedded user question
    "top_k": 5,
})
# Returns ranked list of nearest chunks + metadata + scores
```

Endee's HNSW-based indexing ensures **sub-millisecond search** even over millions of vectors,
making it ideal for real-time RAG applications.

---

## 📁 Project Structure

```
endee-rag/
├── data/                   # Drop your PDFs or .txt files here
├── ingest.py               # Ingestion pipeline
├── query.py                # RAG query engine
├── app.py                  # FastAPI REST API
├── ui.py                   # Streamlit web interface
├── load_sample_data.py     # Creates 3 sample research text files
├── docker-compose.yml      # Spins up Endee locally
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Execution

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- An OpenAI API key (or any OpenAI-compatible LLM endpoint)

---

### Step 1 — Clone the repo
```bash
git clone https://github.com/<your-username>/endee-rag.git
cd endee-rag
```

### Step 2 — Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 3 — Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Start Endee with Docker
```bash
docker-compose up -d
# Endee is now running at http://localhost:8080
```

### Step 5 — Add documents & ingest
```bash
# Option A: Use the included sample research papers
python load_sample_data.py

# Option B: Copy your own PDFs into ./data/
cp /path/to/paper.pdf data/

# Run ingestion
python ingest.py --data-dir data
```

### Step 6 — Start the API server
```bash
python app.py
# FastAPI running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Step 7 — Launch the UI
```bash
streamlit run ui.py
# Opens http://localhost:8501
```

---

## 🧪 Example Queries

Once the system is running with the sample data, try:

- *"What is the attention mechanism in transformers?"*
- *"How does BERT's masked language model work?"*
- *"What is the difference between RAG and fine-tuning?"*
- *"What are the results reported for the Transformer on WMT 2014?"*

---

## 🌐 API Reference

### `POST /ask`
```json
{
  "question": "What is multi-head attention?"
}
```
**Response:**
```json
{
  "answer": "Multi-head attention allows the model to ... [1][2]",
  "sources": [
    { "rank": 1, "title": "attention_is_all_you_need", "page": 3, "score": 0.9241 },
    { "rank": 2, "title": "bert_paper", "page": 1, "score": 0.8873 }
  ]
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for LLM answer generation |
| `ENDEE_URL` | `http://localhost:8080` | Endee server URL |

Chunking and model parameters can be adjusted at the top of `ingest.py` and `query.py`:

```python
CHUNK_SIZE    = 400    # words per chunk
CHUNK_OVERLAP = 50     # word overlap between chunks
EMBED_MODEL   = "all-MiniLM-L6-v2"
TOP_K         = 5      # documents retrieved per query
LLM_MODEL     = "gpt-4o-mini"
```

---

## 🚀 Why Endee?

| Feature | Endee | Typical Alternatives |
|---------|-------|----------------------|
| Deployment | Self-hosted, single binary | Cloud-only or complex K8s setups |
| Scale | Up to 1B vectors on one node | Requires sharding at lower scale |
| Performance | HNSW-based, sub-ms latency | Varies widely |
| Cost | Free / open source | Often expensive at scale |
| API | Simple REST HTTP | Complex gRPC or SDKs |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Endee](https://github.com/endee-io/endee) by endee-io
- [sentence-transformers](https://www.sbert.net/) by UKPLab
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
