# Changelog

All notable changes to **LongParser** are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — planned 0.1.6

### Added

- Whole-project regression tests for HTTP API/review/chat, database, workers, queues, embedding/vector adapters, integrations and conversion/OCR orchestration with controlled external SDKs.

- Extractor classification, heading/hierarchy, page-validation, optional-backend guard and LaTeX OCR setup tests, previously verified without model dependencies on Python 3.10–3.13 before the minimum supported version was raised.
- Offline pipeline, PII redaction, reference, summary and language/RTL/OCR regression tests, with a separate minimal-dependency verification environment and fresh-process default-routing isolation check.
- Controlled regression tests for hybrid chunking, provenance, quality scoring and semantic boundary edge cases, using offline external-resource stubs.
- Opt-in CPU embedding tests with a pinned local MiniLM snapshot and fixed labelled examples; regular CI does not require model assets.

### Changed

- Require Python 3.11 or newer and test supported Python 3.11–3.13 in CI. Python 3.10 support is discontinued following its October 1, 2026 end of life; Python 3.10 users must upgrade to install this upcoming release. This is a compatibility-policy change, not a fix for the unresolved Python 3.10 CI failure.

- Raised the whole-production coverage minimum to 95% after the expanded suite reaches 95.97%; the earlier milestone checkpoint was 47.53% with a 45% minimum. Full CI matrix validation remains pending.
- CI now enforces all configured Ruff rules, with reviewed import and annotation cleanup.
- FastAPI dependency defaults retain their existing behavior through a targeted `B008` exception for the server application.
- The license check detects bare and dotted Marker imports in core files while retaining the two isolated extractor exceptions.
- Added regression tests that execute the actual license-check workflow script and verify lightweight imports in a fresh Python process.
- Coverage now includes every production module; the initial 18.96% baseline used a provisional 17% minimum before the completed milestones raised the gate to 45%.
- Added non-blocking informational mypy checks to CI and documented the coverage measurement policy.
- Removed the unused, empty server routers package and its stale documentation reference; API routes remain in `app.py`.

### Fixed

- Search filters preserve authenticated tenant and selected job scope even when caller filters contain those keys.
- Docling image export initializes and captures the latest successful extraction result; smart formula mode initializes OCR before math detection when no formula blocks are found.

- Wide-table column groups retain every selected value in both row-record and pipe formats by mapping selected columns to local indexes.

- PyMuPDF extraction can now instantiate by implementing the required single-page method, preserving selected-page dimensions and source references and reporting invalid page requests clearly.
- Automatic native-PDF routing now falls back to Docling when the PyMuPDF constructor reports a missing optional dependency, including normal OCR language resolution.
- Equation context carried into another chunk now retains its source page, and outgoing chunks report only the pages of their remaining source blocks.
- Optional-backend installation errors now explicitly retain their original exception causes.
- Vector insertion explicitly uses `zip(strict=False)` to preserve existing truncation behavior.

---

## [0.1.5] — 2026-05-05

### Added

- **Semantic chunking** — `all-MiniLM-L6-v2` embedding-based boundary detection in `HybridChunker` (optional via `use_semantic_chunking`).
- **Cross-reference resolution** — Highly efficient $O(N)$ resolution for explicit ("Figure 3") and implicit ("the table below") references via spatial proximity.
- **Summary chunks** — Asynchronous ARQ background worker (`enrich_summaries_job`) to auto-generate LLM section summaries for hierarchical RAG retrieval.
- **Chunk quality scorer** — Zero-ML, heuristic-based chunk scoring using block token confidences, Dictionary Word Coverage (`/usr/share/dict/words`), and fastText Lang-ID validation.
- **PII redaction** — Hybrid approach using fast Regex+Luhn (Emails, Phones, SSNs, CCs, IPs) and optional spaCy NER (`en_core_web_sm`) for names, organizations, and locations. Preserves original values in secure block metadata for HITL.

### Changed

- Bumped `marker-pdf` version support in dependencies.
- Added `ner` optional dependency group (`spacy>=3.7.0`) in `pyproject.toml`.
- Expanded `ChunkingConfig` and `ProcessingConfig` with new semantic, summary, and PII toggle options.
- Marked Phase 1 as officially complete in Roadmap.

---

## [0.1.3] — 2026-04-13

### Fixed

- **Source code**: Added `DocumentPipeline` as a public alias for `PipelineOrchestrator` —
  docs, quickstart, and all examples now use this name consistently
- **Documentation**: Fixed wrong coverage path `long_parser` → `longparser` in `CONTRIBUTING.md`
- **Documentation**: Replaced stale `cleanrag-api` reference in Docker deployment docs
- **Documentation**: Standardized Gemini API key env var to `GOOGLE_API_KEY` across all docs
- **Source code**: Updated default LLM model fallback from `gpt-4o` to `gpt-5.3` in
  `schemas.py`, `llm_chain.py`, and `engine.py`
- **Source code**: Renamed stale `cleanrag:` Redis key prefix to `longparser:` in embeddings

### Changed

- Python 3.13 added to CI matrix, badges, and installation docs
- `SECURITY.md` updated with Redis rate-limiting and CORS threat mitigations

---

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
- **Multi-provider LLM support** — OpenAI (`gpt-4o`), Gemini (`gemini-2.0-flash`),
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
