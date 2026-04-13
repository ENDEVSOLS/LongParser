<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongParser/main/docs/assets/logo.png" alt="LongParser" width="320">
  <p align="center"><strong>Privacy-first document intelligence engine for production RAG pipelines.</strong></p>
  <p align="center">
    Parse PDFs, DOCX, PPTX, XLSX &amp; CSV → validated, AI-ready chunks with HITL review.
  </p>
  <p align="center">
    <a href="https://github.com/ENDEVSOLS/LongParser/actions/workflows/ci.yml">
      <img src="https://github.com/ENDEVSOLS/LongParser/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>
    <a href="https://pypi.org/project/longparser/">
      <img src="https://img.shields.io/pypi/v/longparser.svg?label=pypi&color=0078d4" alt="PyPI">
    </a>
    <a href="https://pepy.tech/project/longparser">
      <img src="https://static.pepy.tech/badge/longparser" alt="Total Downloads">
    </a>
    <a href="https://pepy.tech/project/longparser">
      <img src="https://static.pepy.tech/badge/longparser/month" alt="Monthly Downloads">
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="MIT License">
    </a>
    <a href="https://endevsols.github.io/LongParser/">
      <img src="https://img.shields.io/badge/docs-online-indigo.svg" alt="Docs">
    </a>
  </p>
</p>

---


## Features

| Feature | Detail |
|---------|--------|
| **Multi-format extraction** | PDF, DOCX, PPTX, XLSX, CSV via Docling |
| **Hybrid chunking** | Token-aware, heading-hierarchy-aware, table-aware |
| **HITL review** | Human-in-the-Loop block & chunk editing before embedding |
| **LangGraph HITL** | `approve / edit / reject` workflow with LangGraph `interrupt()` and MongoDB checkpointer |
| **3-layer memory** | Short-term turns + rolling summary + long-term facts |
| **Multi-provider LLM** | OpenAI, Gemini, Groq, OpenRouter |
| **Multi-backend vectors** | Chroma, FAISS, Qdrant |
| **Production-ready API** | FastAPI + Motor (MongoDB) + ARQ + Redis (Queue & Rate Limiting) |
| **Enterprise Security** | Tenant isolation, Role-Based Access Control (RBAC), and CORS |
| **LangChain adapters** | Drop-in `BaseRetriever` and LlamaIndex `QueryEngine` |
| **Privacy-first** | All processing runs locally; no data leaves your infra |

---

## Installation

### Quick install (recommended)

```bash
pip install "longparser[gpu]"
```

Includes everything — server, embeddings, vector DB, OCR, LangChain, LlamaIndex.
Works on CPU machines too; torch just runs in CPU mode automatically.

### Core SDK only (no server, no torch)

```bash
pip install longparser
```

### Pick only what you need

| Extra | What it adds |
|---|---|
| `server` | FastAPI + MongoDB + Redis + LangChain chat |
| `embeddings-gpu` | `sentence-transformers` (GPU) |
| `embeddings-cpu` | `sentence-transformers` (CPU-only torch) |
| `faiss-gpu` | FAISS GPU vector store |
| `faiss-cpu` | FAISS CPU vector store |
| `chroma` | ChromaDB |
| `qdrant` | Qdrant |
| `latex-ocr-gpu` | `pix2tex` equation OCR (GPU) |
| `latex-ocr-cpu` | `pix2tex` equation OCR (CPU) |
| `langchain` | LangChain core adapter |
| `llamaindex` | LlamaIndex reader adapter |
| `gpu` | **All of the above** — one command |
| `cpu` | **All of the above** — CPU-only torch |

### Advanced: CPU-only install (save ~1.8 GB)

For Docker images, edge devices, or CI environments where CUDA isn't needed:

```bash
# Step 1 — CPU torch (~230 MB vs ~2 GB for CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2 — LongParser CPU bundle
pip install "longparser[cpu]"
```

---


## Quick Start

### Python SDK

```python
from longparser import DocumentPipeline, ProcessingConfig

pipeline = DocumentPipeline(ProcessingConfig())
result = pipeline.process_file("document.pdf")

print(f"Pages: {result.document.metadata.total_pages}")
print(f"Chunks: {len(result.chunks)}")
print(result.chunks[0].text)
```

### REST API

```bash
# 1. Copy and edit configuration
cp .env.example .env

# 2. Start services (MongoDB + Redis)
docker-compose up -d mongo redis

# 3. Start the API
uv run uvicorn longparser.server.app:app --reload --port 8000

# 4. Upload a document
curl -X POST http://localhost:8000/jobs \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf"

# 5. Check job status
curl http://localhost:8000/jobs/{job_id} -H "X-API-Key: your-key"

# 6. Finalize and embed
curl -X POST http://localhost:8000/jobs/{job_id}/finalize \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"finalize_policy": "approve_all_pending"}'

curl -X POST http://localhost:8000/jobs/{job_id}/embed \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"provider": "huggingface", "model": "BAAI/bge-base-en-v1.5", "vector_db": "chroma"}'

# 7. Chat with the document
curl -X POST http://localhost:8000/chat/sessions \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "your-job-id"}'

curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "job_id": "...", "question": "What is the refund policy?"}'
```

---

## Architecture

```
Document → Extract → Validate → HITL Review → Chunk → Embed → Index
                                                              ↓
                                             Chat → RAG → LLM → Answer
```

### Pipeline Stages

1. **Extract** — Docling converts PDF/DOCX/etc. into structured `Block` objects
2. **Validate** — Per-page confidence scoring and RTL detection
3. **HITL Review** — Human approves/edits/rejects blocks and chunks via the API
4. **Chunk** — `HybridChunker` builds token-aware RAG chunks with section hierarchy
5. **Embed** — Embedding engine (HuggingFace / OpenAI) vectors stored in Chroma/FAISS/Qdrant
6. **Chat** — LCEL chain with 3-layer memory and citation validation

---

## Project Structure

```
src/longparser/
├── schemas.py           ← core Pydantic models (Document, Block, Chunk, …)
├── extractors/          ← Docling, LaTeX OCR backends
├── chunkers/            ← HybridChunker
├── pipeline/            ← DocumentPipeline
├── integrations/        ← LangChain loader & LlamaIndex reader
├── utils/               ← shared helpers (RTL detection, …)
└── server/              ← REST API layer
    ├── app.py           ← FastAPI application (all routes)
    ├── db.py            ← Motor async MongoDB
    ├── queue.py         ← ARQ/Redis job queue
    ├── worker.py        ← ARQ background worker
    ├── embeddings.py    ← HuggingFace / OpenAI embedding engine
    ├── vectorstores.py  ← Chroma / FAISS / Qdrant adapters
    └── chat/            ← RAG chat engine
        ├── engine.py    ← ChatEngine (LCEL + 3-layer memory)
        ├── graph.py     ← LangGraph HITL workflow
        ├── schemas.py   ← chat Pydantic models
        ├── retriever.py ← LangChain BaseRetriever adapter
        ├── llm_chain.py ← multi-provider LLM factory
        └── callbacks.py ← observability callbacks
```

---

## LangChain Integration

```python
from longparser.integrations.langchain import LongParserLoader

loader = LongParserLoader("report.pdf")
docs = loader.load()  # list[langchain_core.documents.Document]
```

## LlamaIndex Integration

```python
from longparser.integrations.llamaindex import LongParserReader

reader = LongParserReader()
docs = reader.load_data("report.pdf")
```

---

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Default | Description |
|----------|---------|-------------|
| `LONGPARSER_MONGO_URL` | `mongodb://localhost:27017` | MongoDB connection |
| `LONGPARSER_REDIS_URL` | `redis://localhost:6379` | Redis for job queue & rate limits |
| `LONGPARSER_LLM_PROVIDER` | `openai` | LLM provider |
| `LONGPARSER_LLM_MODEL` | `gpt-5.3` | Model name |
| `LONGPARSER_EMBED_PROVIDER` | `huggingface` | Embedding provider |
| `LONGPARSER_VECTOR_DB` | `chroma` | Vector store backend |
| `LONGPARSER_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `LONGPARSER_RATE_LIMIT` | `60` | Max RPM per tenant |
| `LONGPARSER_ADMIN_KEYS` | (empty) | Comma-separated admin API keys |

---

## Running with Docker

```bash
docker-compose up
```

API available at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

---

## Testing

```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/ --cov=src/longparser --cov-report=term-missing
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

[MIT](LICENSE) — Copyright © 2026 ENDEVSOLS
