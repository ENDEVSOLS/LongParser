# Changelog

All notable changes to **LongParser** are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.2] — 2026-04-05

### Changed

- Project logo added to documentation site, README, and PyPI page
- Documentation site header updated — logo replaces text title
- Installation guide restructured for clarity

---

## [0.1.1] — 2026-04-04

### Added

- **CPU / GPU install separation** — dedicated `[cpu]` and `[gpu]` meta-extras for clean one-command installs
- **`faiss-gpu`** extra (`faiss-gpu>=1.7`) as a distinct option from `faiss-cpu`
- **Granular torch-based extras** — `embeddings-cpu`, `embeddings-gpu`, `latex-ocr-cpu`, `latex-ocr-gpu` for fine-grained dependency control

### Fixed

- Package metadata: license field updated to SPDX expression format per PEP 639
- Documentation site build reliability improvements

### Changed

- `[gpu]` is now the recommended default install — one command, works on both GPU and CPU machines
- `[cpu]` documented as the advanced path for size-constrained environments (Docker, edge, CI)
- `[all]` now resolves to `[cpu]` as a safe, dependency-minimal default

---


## [0.1.0] — 2026-04-04


### 🎉 Initial Public Release

LongParser is the open-source document intelligence engine built by ENDEVSOLS
for production RAG pipelines.

### Added

- **5-stage extraction pipeline** — `Extract → Validate → HITL Review → Chunk → Embed → Index`
- **Multi-format extraction** — PDF, DOCX, PPTX, XLSX, CSV via Docling
- **`HybridChunker`** — token-aware, heading-hierarchy-aware, table-aware chunking
- **Human-in-the-Loop (HITL) review** — approve / edit / reject blocks and chunks
  via LangGraph `interrupt()` before embedding
- **3-layer memory chat** — short-term turns + rolling summary + long-term facts,
  powered by LCEL chains
- **Multi-provider LLM support** — OpenAI (`gpt-5.3`), Gemini (`gemini-2.5`),
  Groq (`llama-3.3-70b-versatile`), OpenRouter
- **Multi-backend vector stores** — Chroma, FAISS, Qdrant
- **Async-first REST API** — FastAPI + Motor (MongoDB) + ARQ (Redis job queue)
- **`LongParserRetriever`** — drop-in LangChain `BaseRetriever` adapter
- **`LongParserLoader`** — LangChain document loader integration
- **`LongParserReader`** — LlamaIndex `BaseReader` integration
- **`LongParserCallbackHandler`** — observability callbacks for LangChain chains
- **Built-in citation validation** — chunk IDs verified against retrieved set
  before any answer is returned
- **Privacy-first** — all processing runs locally; no data leaves your infrastructure
- **`py.typed` marker** — full PEP 561 typing support
- **Unit test suite** — `test_schemas.py` (22 passing), `test_llm_chain.py`,
  `test_chat_utils.py`
- **GitHub Actions CI** — lint (`ruff`), tests across Python 3.10 / 3.11 / 3.12,
  coverage reporting
- **GitHub Actions publish** — PyPI trusted publishing triggered on GitHub releases
- **`pyproject.toml`** with `server`, `langchain`, `llamaindex`, `embeddings`,
  `chroma`, `faiss`, `qdrant` optional extras
- **`Dockerfile`** and **`docker-compose.yml`** for one-command local deployment
- **`CONTRIBUTING.md`**, **`SECURITY.md`**, **`.env.example`** — full OSS scaffolding

