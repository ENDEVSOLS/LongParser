# LongParser — Product & Feature Roadmap

> This roadmap reflects the current development direction based on community trends,
> competitor analysis, and the RAG ecosystem in 2025–2026. Items are ordered by
> priority within each phase. All dates are targets, not guarantees.

---

## Current State — v0.1.x ✅

- 5-stage extraction pipeline (Extract → Validate → HITL → Chunk → Embed → Index)
- Multi-format support: PDF, DOCX, PPTX, XLSX, CSV via Docling
- `HybridChunker` — 6-strategy token-aware, hierarchy-aware, table-aware chunking
- Human-in-the-Loop (HITL) review via LangGraph `interrupt()`
- 3-layer memory chat engine (short-term + rolling summary + long-term facts)
- Multi-provider LLM: OpenAI, Gemini, Groq, OpenRouter
- Multi-backend vector stores: Chroma, FAISS, Qdrant
- FastAPI REST server + ARQ/Redis job queue + Motor/MongoDB
- LangChain `BaseRetriever` + LlamaIndex `BaseReader` adapters
- CPU / GPU install separation via extras

---

## Phase 1 — Accuracy & Quality (v0.2.x) — Q2 2026

### Parser Enhancements

- [ ] **Marker backend** — add `marker-pdf` as an optional extraction backend for higher-fidelity Markdown output on complex academic PDFs
- [ ] **PyMuPDF4LLM backend** — lightweight, fast alternative for speed-critical pipelines (10× faster than Docling for simple PDFs)
- [ ] **Scanned PDF fast path** — route documents to Tesseract vs pix2tex vs Surya automatically based on page complexity score
- [ ] **Multi-column layout detection** — prevent reading-order errors in newspaper/journal-style layouts
- [ ] **Image extraction** — export embedded figures with captions into separate chunks with `type: figure`
- [ ] **Document language auto-detection** — select OCR model automatically based on detected script

### Chunking Improvements

- [ ] **Semantic chunking** — optional embedding-based boundary detection (split at semantic shifts, not just token counts)
- [ ] **Sliding window overlap** — configurable overlap strategy per chunk type (more overlap for tables, less for headings)
- [ ] **Cross-reference resolution** — link `(see Figure 3)` and `(Table 2)` references to their target blocks
- [ ] **Summary chunks** — auto-generate a 1–2 sentence summary chunk per section for hierarchical retrieval

### Quality & Validation

- [ ] **Chunk quality scorer** — assign a confidence score per chunk based on OCR confidence, completeness, and structural integrity
- [ ] **PII detection** — flag and optionally redact personal information (names, emails, phone numbers) before embedding
- [ ] **Duplicate block detection** — suppress repeated headers/footers that appear on every page

---

## Phase 2 — Agentic & Multimodal (v0.3.x) — Q3 2026

### Agentic RAG

- [ ] **Agentic retrieval loop** — implement query rewriting + iterative retrieval + self-reflection before answer generation
- [ ] **Multi-hop question answering** — chain retrieval steps for questions that span multiple sections or documents
- [ ] **Tool-calling integration** — expose document pipeline as a LangChain/LangGraph tool callable by autonomous agents
- [ ] **Hypothetical Document Embeddings (HyDE)** — generate hypothetical answers to queries for improved retrieval recall

### Multimodal

- [ ] **Vision-Language Model (VLM) integration** — use GPT-4o / Gemini Vision to describe figures, charts, and diagrams as text chunks
- [ ] **Chart data extraction** — parse bar/line/pie charts into structured data tables
- [ ] **Slide layout understanding** — treat PPTX slides as visual units with spatial layout context, not just text extraction

### Reranking & Retrieval

- [ ] **Cross-encoder reranker** — add optional `sentence-transformers` cross-encoder reranking step after initial retrieval
- [ ] **Hybrid search** — combine dense vector search with BM25 sparse retrieval (reciprocal rank fusion)
- [ ] **Maximum Marginal Relevance (MMR)** — reduce redundancy in retrieved chunks
- [ ] **Metadata filtering** — filter chunks by `page_number`, `section`, `doc_type`, `date` at query time

---

## Phase 3 — Enterprise & Observability (v0.4.x) — Q4 2026

### Knowledge Graph

- [ ] **Entity extraction** — extract named entities (people, organizations, dates, locations) from chunks
- [ ] **Relationship mapping** — build entity relationship graphs from document content
- [ ] **Graph-based retrieval** — traverse the entity graph for multi-hop retrieval (GraphRAG pattern)
- [ ] **Neo4j / NetworkX integration** — persist the knowledge graph to a graph database

### Evaluation Framework

- [ ] **Built-in RAG evaluator** — measure retrieval recall@k, answer faithfulness, and context adherence
- [ ] **Chunk attribution** — trace every answer sentence back to the source chunk and page
- [ ] **RAGAS integration** — plug into the RAGAS evaluation framework
- [ ] **Benchmark suite** — reproducible benchmarks against Unstructured, LlamaParse, Docling standalone

### Observability & Compliance

- [ ] **LangSmith integration** — trace every pipeline run end-to-end
- [ ] **OpenTelemetry support** — emit spans/traces to any OTel-compatible backend
- [ ] **Audit log** — immutable log of every HITL decision (approve/reject/edit) with timestamps and user IDs
- [ ] **GDPR compliance mode** — PII redaction + right-to-erasure support (delete all chunks for a document)
- [ ] **Role-based access control (RBAC)** — multi-tenant document access in the REST API

---

## Phase 4 — Scale & Ecosystem (v0.5.x+) — 2027

### Performance & Scale

- [ ] **Async parallel extraction** — process multiple documents concurrently in the background worker
- [ ] **Streaming extraction** — yield blocks as they are extracted (no need to wait for full document)
- [ ] **Incremental indexing** — update only changed pages/sections on re-upload
- [ ] **S3 / GCS / Azure Blob** — native cloud storage input (not just local files)
- [ ] **Kubernetes Helm chart** — one-command production deployment

### New Integrations

- [ ] **Weaviate** vector store adapter
- [ ] **Pinecone** vector store adapter
- [ ] **Milvus** vector store adapter
- [ ] **DSPy** integration — use DSPy to auto-optimize retrieval prompts
- [ ] **Haystack `DocumentConverter`** component
- [ ] **Flowise / Langflow** node — drag-and-drop visual pipeline builder support

### Developer Experience

- [ ] **LongParser CLI** — `longparser parse document.pdf --output chunks.json`
- [ ] **Web UI (HITL Dashboard)** — visual interface for reviewing and editing blocks before embedding
- [ ] **VS Code extension** — preview parsed chunks directly from the editor
- [ ] **Webhook support** — notify external systems when a job completes or requires HITL review

---

## Competitive Positioning

| Capability | LongParser | Unstructured | LlamaParse | Docling |
|---|---|---|---|---|
| Privacy-first (fully local) | ✅ | ⚠️ (cloud option) | ❌ (API-only) | ✅ |
| HITL review workflow | ✅ | ❌ | ❌ | ❌ |
| Bundled REST API server | ✅ | ✅ (paid) | ✅ (cloud) | ❌ |
| Table-aware chunking | ✅ | ⚠️ | ✅ | ✅ |
| LaTeX / equation OCR | ✅ | ❌ | ⚠️ | ⚠️ |
| LangChain + LlamaIndex | ✅ | ✅ | ✅ | ⚠️ |
| Open source (MIT) | ✅ | ⚠️ (core only) | ❌ | ✅ |
| Knowledge graph (planned) | 🔜 | ❌ | ❌ | ❌ |
| Agentic retrieval (planned) | 🔜 | ❌ | ⚠️ | ❌ |

---

## Guiding Principles

1. **Privacy by default** — all processing runs locally; no data leaves user infrastructure
2. **Human oversight** — HITL is a first-class citizen, not an afterthought
3. **Composable** — every stage is independently usable; no forced lock-in to the full stack
4. **Production-grade** — async, typed, tested, documented from day one
5. **Ecosystem-native** — LangChain, LlamaIndex, and HuggingFace are first-class integration targets
