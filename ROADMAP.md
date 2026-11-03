# LongParser Roadmap

This document outlines the version milestones, completed capabilities, and future directions for **LongParser**.

---

## Milestone Overview

| Version | Focus | Status |
|---|---|---|
| **v0.1.0 – v0.1.5** | **Phase 1: Core Document Intelligence Engine**<br>Multi-format extraction (PDF, DOCX, PPTX, XLSX, CSV), `HybridChunker`, semantic chunking, cross-references, PII redaction, 3-layer memory chat, HITL review, and REST API. | **Completed** |
| **v0.2.0** | **Trustworthy Core (Hardening)**<br>Comprehensive test suite covering all modules (>95% coverage), full Ruff lint enforcement, empty routers package removal, license safety gate, and formalization of the optional `MarkerExtractor` backend. | **Delivered** |
| **v0.3.0** | **Phase 2: Advanced Document Intelligence & Enhancements**<br>Maintainer feature enhancements, expanded OCR routing, and extended format pre-processors. | **Planned** |
| **v0.4.0+** | **Modular Server Architecture & Performance Baseline**<br>Modular decomposition of server routes, benchmarked performance telemetry, and optional `surya` OCR support. | **Future** |
| **v1.0.0** | **Enterprise API Stability & Production Validation**<br>Strict API stability guarantees, backward compatibility policies, and production deployment validations. | **Future** |

---

## Detailed Milestone Scope

### Phase 1: Core Foundation (v0.1.0 – v0.1.5) — Completed
- **Multi-Format Extraction:** Docling integration for parsing PDF, DOCX, PPTX, XLSX, and CSV documents into unified schemas.
- **Hierarchical Chunking:** `HybridChunker` with token awareness, table structure preservation, and heading hierarchy tracking.
- **Semantic Chunking:** Boundary detection using `all-MiniLM-L6-v2` embeddings.
- **Quality & Privacy:** Zero-ML chunk quality scoring heuristics and hybrid regex/spaCy PII redaction.
- **Enterprise Integrations:** LangChain and LlamaIndex retriever and query engine adapters.
- **REST & Memory Services:** FastAPI server, MongoDB checkpointer, ARQ background worker queues, and multi-turn chat memory.

### Phase 2: Trustworthy Core (v0.2.0) — Delivered
- **Comprehensive Unit Tests:** 440+ regression and unit tests covering `chunkers/`, `extractors/`, `pipeline/`, and `server/`.
- **Enforced Code Standards:** Repository-wide enforcement of all configured Ruff lint rules and type safety guards.
- **Strict License Isolation:** Verified isolation of optional AGPL (`pymupdf4llm`) and GPL (`marker-pdf`) backends, ensuring core imports remain pure MIT.
- **Marker Backend Formalization:** Fully exposed `MarkerExtractor` via lazy imports with documentation and licensing notices.
- **Verified Coverage Floor:** Code coverage gate set to 95.0% enforced in CI.
- **Owner-Managed Releases:** Direct maintainer control over release bumping and PyPI publishing.

### Future Horizons (v0.3.0 & Beyond) — Planned
- **Server Decomposition:** Refactor `longparser/server/app.py` into dedicated, modular router packages.
- **Surya Backend:** Formal evaluation and integration of the optional Surya multilingual OCR engine under GPL isolation.
- **Performance Telemetry:** Published throughput, memory, and latency baselines across document sizes.
- **Production Validation:** Long-running stress testing and multi-tenant isolation benchmarking.

