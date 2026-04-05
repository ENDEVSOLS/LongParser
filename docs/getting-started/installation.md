# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- Tesseract OCR (`brew install tesseract` / `apt install tesseract-ocr`)

---

## Quick Install (recommended)

For most users — includes everything (server, embeddings, vector DB, OCR):

```bash
pip install "longparser[gpu]"
```

> This pulls the standard PyTorch wheel from PyPI which includes CUDA support.
> It works on **CPU machines too** — torch just runs in CPU mode automatically.

---

## Core SDK only (no server, no torch)

Just document parsing and chunking — minimal footprint:

```bash
pip install longparser
```

---

## Pick only what you need

| Extra | Installs | Notes |
|---|---|---|
| `server` | FastAPI + MongoDB + Redis + LangChain chat | REST API layer |
| `embeddings-gpu` | `sentence-transformers` | GPU torch recommended |
| `embeddings-cpu` | `sentence-transformers` | Pre-install CPU torch first |
| `faiss-gpu` | `faiss-gpu` vector store | Included in `[gpu]` |
| `faiss-cpu` | `faiss-cpu` vector store | Included in `[cpu]` |
| `chroma` | ChromaDB vector store | No torch needed |
| `qdrant` | Qdrant vector store | No torch needed |
| `latex-ocr-gpu` | `pix2tex` equation OCR | GPU torch |
| `latex-ocr-cpu` | `pix2tex` equation OCR | CPU torch |
| `docx-equations` | OMML→LaTeX for DOCX/PPTX | No torch needed |
| `langchain` | LangChain core adapter | No torch needed |
| `llamaindex` | LlamaIndex reader adapter | No torch needed |
| `pptx` | python-pptx indent detection | No torch needed |
| `gpu` | **All of the above** (GPU) | ✅ Recommended |
| `cpu` | **All of the above** (CPU-only) | See Advanced below |

```bash
# Example: server + chroma only (no torch needed at all)
pip install "longparser[server,chroma]"

# Example: server + GPU embeddings + qdrant
pip install "longparser[server,embeddings-gpu,qdrant]"
```

---

## Advanced: CPU-only install (minimal size)

Use this if you want to save ~1.8 GB by avoiding the CUDA torch build.
Useful for Docker images, edge devices, or CI environments.

```bash
# Step 1 — install CPU-only torch first (~230 MB vs ~2 GB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2 — install LongParser with CPU extras
pip install "longparser[cpu]"
```

---

## With uv (recommended for development)

```bash
git clone https://github.com/ENDEVSOLS/LongParser.git
cd LongParser

# GPU (default)
uv sync --extra gpu

# CPU-only
uv sync --extra cpu
```

---

## Docker

```bash
docker compose up
```

The server starts on `http://localhost:8000`.

---

## Verify Installation

```python
import longparser
print(longparser.__version__)  # 0.1.2
```
