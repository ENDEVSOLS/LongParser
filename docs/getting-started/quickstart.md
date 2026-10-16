# Quickstart

Extract a document and explicitly create chunks for retrieval.

## 1. Install

```bash
pip install longparser
```

## 2. Parse a PDF

```python
from longparser import ChunkingConfig, DocumentPipeline, ProcessingConfig

# Create pipeline with defaults
config = ProcessingConfig()
pipeline = DocumentPipeline(config)

# Parse a PDF
result = pipeline.process_file("research_paper.pdf", config=config)
chunks = pipeline.chunk(result, ChunkingConfig())

print(f"Pages: {result.document.metadata.total_pages}")
print(f"Chunks: {len(result.chunks)}")
if chunks:
    print(chunks[0].text)
```

`process_file()` extracts blocks and hierarchy; `chunk()` fills `result.chunks`.
Embedding, indexing and retrieval are separate stages. The default PDF OCR path
requires a Tesseract executable and appropriate language data. For documents
that do not need OCR, choose processing options appropriate to the input.

## 3. Inspect Chunks

```python
for chunk in result.chunks[:3]:
    print(f"[{chunk.chunk_type}] tokens={chunk.token_count}")
    print(chunk.text[:200])
    print("---")
```

## 4. Use with LangChain

Install `longparser[langchain]` to use the loader.

```python
from longparser.integrations.langchain import LongParserLoader

loader = LongParserLoader("report.pdf")
documents = loader.load()  # Returns List[Document]
```

## 5. Use with LlamaIndex

Install `longparser[llamaindex]` to use the reader.

```python
from longparser.integrations.llamaindex import LongParserReader

reader = LongParserReader()
nodes = reader.load_data(file="report.pdf")
```

## 6. Start the REST Server

```bash
# Set environment variables
cp .env.example .env
# Edit .env with your keys

# Start the API, background worker, MongoDB and Redis
docker compose up -d --build
```

Then visit [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

## Supported Formats

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | OCR + table structure |
| Word | `.docx` | OMML equation injection |
| PowerPoint | `.pptx` | Slide-by-slide chunking |
| Excel | `.xlsx` | Sheet-aware table parsing |
| CSV | `.csv` | Column-profile chunks |
