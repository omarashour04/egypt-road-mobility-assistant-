# 🚗 Egyptian Road & Mobility Assistant

A bilingual (Arabic/English) Retrieval-Augmented Generation (RAG) system for answering questions about Egyptian traffic laws, driving licenses, vehicle registration, and road rules.

Built as an NLP course project (Project 2 — Website Question Answering).

---

## Architecture

```
User Query
    │
    ▼
Query Enhancer ──── HyDE (hypothetical document embedding)
    │            └── Bilingual translation (AR ↔ EN)
    ▼
FAISS Retriever ─── 3 parallel searches (raw + HyDE + translated)
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
    │
    ▼
Qwen3:4b Generator (via Ollama)
    │
    ▼
Answer + Sources + Session Memory
```

---

## Domains Covered

| Domain | Arabic | English |
|--------|--------|---------|
| Traffic Law | ✅ | ✅ |
| Driving License | ✅ | ✅ |
| Vehicle Registration | ✅ | ✅ |
| Accident Liability | ✅ | ✅ |
| Commercial Vehicles | ✅ | ✅ |
| Driver Fitness | ✅ | ✅ |
| International Driving | ✅ | ✅ |
| Road Infrastructure | ✅ | ✅ |

---

## Setup

### Prerequisites
- Python 3.11
- [Ollama](https://ollama.com) installed and running
- Conda (recommended)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create conda environment
```bash
conda create -n traffic_qa python=3.11
conda activate traffic_qa
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models
```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

### 5. Scrape & index
```bash
python run.py scrape    # Collect data from sources (~15–30 min)
python run.py index     # Build FAISS index (~4 min)
```

### 6. Start the API
```bash
python run.py serve
```

API docs available at **http://localhost:8000/docs** (Swagger UI)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question (Arabic or English) |
| `POST` | `/scrape` | Trigger a fresh scrape |
| `GET` | `/health` | System health check |
| `GET` | `/stats` | Index statistics |
| `GET` | `/session/{id}/history` | Conversation history |
| `DELETE` | `/session/{id}` | Clear a session |

### Example request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي غرامة تجاوز السرعة في مصر؟"}'
```

### Example response
```json
{
  "answer": "غرامة تجاوز السرعة في مصر تتراوح بين 500 و3000 جنيه...",
  "sources": ["youm7.com", "masrawy.com"],
  "session_id": "abc123",
  "language_detected": "ar"
}
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `qwen3-embedding:0.6b` via Ollama |
| Generation | `qwen3:4b` via Ollama |
| Vector Store | FAISS (CPU) |
| Reranker | `ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| API | FastAPI + Swagger UI |
| Scraping | BeautifulSoup + requests |

---

## Project Structure

```
├── config.py               # All configuration (sources, model names, paths)
├── run.py                  # CLI entrypoint: scrape / index / serve / eval
├── requirements.txt
├── scraper/
│   ├── scraper.py          # Web scraping with retry + rate limiting
│   └── parser.py           # HTML parsing, cleaning & sentence chunking
├── indexer/
│   ├── embedder.py         # Batch embedding generation via Ollama
│   └── faiss_store.py      # FAISS IndexFlatIP build & search
├── rag/
│   ├── query_enhancer.py   # HyDE + bilingual AR↔EN translation
│   ├── retriever.py        # Orchestrates 3-way FAISS search + dedup
│   ├── reranker.py         # Cross-encoder reranking (ms-marco-MiniLM)
│   └── generator.py        # Prompt construction + Qwen3 generation
├── api/
│   ├── main.py             # FastAPI app & route definitions
│   ├── pipeline.py         # End-to-end RAG pipeline orchestration
│   ├── schemas.py          # Pydantic request/response models
│   └── session.py          # Conversation memory with 30-min TTL
└── eval/
    └── evaluate.py         # ROUGE-L + retrieval recall across 8 domains
```

---

## Running Evaluation

```bash
# Baseline (no HyDE, no reranking)
python run.py eval --no-hyde --no-rerank

# Full pipeline
python run.py eval
```
