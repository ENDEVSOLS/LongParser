# Configuration

LongParser is configured entirely via environment variables (no config files to manage).

## Core Variables

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

## Required Variables

| Variable | Description |
|---|---|
| `LONGPARSER_API_KEY` | API key for the REST server |
| `LONGPARSER_MONGO_URL` | MongoDB connection string |
| `OPENAI_API_KEY` | For OpenAI LLM provider |

## Processing Options

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_FORMULA_MODE` | `smart` | `fast` / `smart` / `full` |
| `LONGPARSER_MAX_TOKENS` | `512` | Max tokens per chunk |
| `LONGPARSER_CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `LONGPARSER_UPLOAD_DIR` | `./uploads` | Upload directory |

## LLM Providers

| Variable | Description |
|---|---|
| `LONGPARSER_LLM_PROVIDER` | `openai` / `gemini` / `groq` / `openrouter` |
| `LONGPARSER_LLM_MODEL` | Model name (uses provider default if unset) |
| `GEMINI_API_KEY` | For Google Gemini |
| `GROQ_API_KEY` | For Groq |

## Vector Store

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_VECTOR_DB` | `chroma` | `chroma` / `faiss` / `qdrant` |
| `LONGPARSER_COLLECTION` | `longparser` | Collection name |
| `QDRANT_URL` | — | Qdrant server URL (if using Qdrant) |

## ProcessingConfig Defaults

When using the Python SDK directly, configure via `ProcessingConfig`:

```python
from longparser import ProcessingConfig

config = ProcessingConfig(
    do_ocr=True,
    do_table_structure=True,
    formula_mode="smart",       # fast | smart | full
    formula_ocr=True,
    export_images=False,
    max_pages=None,             # None = all pages
)
```

## ChunkingConfig Defaults

```python
from longparser.schemas import ChunkingConfig

config = ChunkingConfig(
    max_tokens=512,
    overlap_tokens=64,
    detect_equations=True,
    exclude_headers_footers=True,
    generate_schema_chunks=True,    # table schema chunks
    table_chunk_format="row_record", # pipe | row_record
    wide_table_col_threshold=15,
)
```
