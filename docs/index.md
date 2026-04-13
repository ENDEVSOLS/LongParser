<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongParser/main/docs/assets/logo.png" alt="LongParser" width="320">
  <p align="center"><strong>Privacy-first document intelligence engine for production RAG pipelines.</strong></p>
  <p align="center">Parse PDFs, DOCX, PPTX, XLSX &amp; CSV → validated, AI-ready chunks with HITL review.</p>
  <p align="center">
    <a href="https://github.com/ENDEVSOLS/LongParser/actions/workflows/ci.yml">
      <img src="https://github.com/ENDEVSOLS/LongParser/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>&nbsp;
    <a href="https://pypi.org/project/longparser/">
      <img src="https://img.shields.io/pypi/v/longparser.svg?label=pypi&color=0078d4" alt="PyPI">
    </a>&nbsp;
    <a href="https://pepy.tech/project/longparser">
      <img src="https://static.pepy.tech/badge/longparser" alt="Total Downloads">
    </a>&nbsp;
    <a href="https://pepy.tech/project/longparser">
      <img src="https://static.pepy.tech/badge/longparser/month" alt="Monthly Downloads">
    </a>&nbsp;
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python">
    </a>&nbsp;
    <a href="https://github.com/ENDEVSOLS/LongParser/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="MIT License">
    </a>
  </p>
</p>

---


## Why LongParser?

Most RAG pipelines fail at the data layer. Hallucinations, missed tables, garbled equations, and unverified citations stem from poor document parsing — not from the LLM itself.

**LongParser solves the input problem.**

| Feature | LongParser |
|---|---|
| Multi-format extraction | PDF, DOCX, PPTX, XLSX, CSV |
| Hybrid chunking (6 strategies) | ✅ |
| HITL review workflow | ✅ |
| 3-layer memory chat | ✅ |
| Built-in citation validation | ✅ |
| LaTeX/equation parsing | ✅ |
| LangChain & LlamaIndex ready | ✅ |
| RTL language support | ✅ |
| Docker-ready server | ✅ |

---

## Quick Start

```bash
pip install longparser
```

```python
from longparser import DocumentPipeline, ProcessingConfig

pipeline = DocumentPipeline(ProcessingConfig())
result = pipeline.process_file("report.pdf")

print(f"Chunks: {len(result.chunks)}")
print(result.chunks[0].text)
```

---

## Architecture

```mermaid
graph LR
    A[Document] --> B[Extract]
    B --> C[Validate]
    C --> D[HITL Review]
    D --> E[Chunk]
    E --> F[Embed]
    F --> G[Index]
    G --> H[Chat Engine]
```

---

## Installation

```bash
# Recommended — everything included (GPU/CPU both work)
pip install "longparser[gpu]"

# Core SDK only — minimal, no server
pip install longparser
```

→ [Full installation guide](getting-started/installation.md) — CPU-only, Docker, extras reference


## Next Steps

- [**Installation Guide**](getting-started/installation.md) — detailed setup with virtual environments
- [**Quickstart**](getting-started/quickstart.md) — parse your first document in 5 minutes
- [**Configuration**](getting-started/configuration.md) — environment variables and tuning
