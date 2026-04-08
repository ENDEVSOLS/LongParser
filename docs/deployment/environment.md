# Environment Variables

Copy `.env.example` to `.env` and configure for your deployment.

## Required

| Variable | Description |
|---|---|
| `LONGPARSER_API_KEY` | API key for server authentication |
| `LONGPARSER_MONGO_URL` | MongoDB connection string |

## LLM

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_LLM_PROVIDER` | `openai` | LLM provider |
| `LONGPARSER_LLM_MODEL` | _(provider default)_ | Model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |

## Processing

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_UPLOAD_DIR` | `./uploads` | Upload directory path |
| `LONGPARSER_FORMULA_MODE` | `smart` | `fast` / `smart` / `full` |
| `LONGPARSER_LATEX_OCR_BACKEND` | `pix2tex` | LaTeX OCR backend |
| `LONGPARSER_FORMULA_PER_EQ_TIMEOUT` | `30` | Per-equation OCR timeout (seconds) |

## Chunking

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_MAX_TOKENS` | `512` | Max tokens per chunk |
| `LONGPARSER_CHUNK_OVERLAP` | `64` | Overlap between chunks |

## Vector Store

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_VECTOR_DB` | `chroma` | `chroma` / `faiss` / `qdrant` |
| `LONGPARSER_COLLECTION` | `longparser` | Default collection name |
| `QDRANT_URL` | — | Qdrant server URL |

## Infrastructure

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_REDIS_URL` | `redis://localhost:6379/0` | Redis URL for task queue |
| `LONGPARSER_WORKER_CONCURRENCY` | `2` | Worker concurrency level |

## Security

| Variable | Default | Description |
|---|---|---|
| `LONGPARSER_CORS_ORIGINS` | `*` | Allowed CORS origins (comma separated) |
| `LONGPARSER_RATE_LIMIT` | `60` | Max requests per minute per tenant ID |
| `LONGPARSER_ADMIN_KEYS` | — | Comma-separated admin API keys |
