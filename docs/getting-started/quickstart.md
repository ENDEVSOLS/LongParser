# Quickstart

Parse your first document in under 5 minutes.

## 1. Install

```bash
pip install longparser
```

## 2. Parse a PDF

```python
from longparser import DocumentPipeline, ProcessingConfig

# Create pipeline with defaults
pipeline = DocumentPipeline(ProcessingConfig())

# Parse a PDF
result = pipeline.process_file("research_paper.pdf")

print(f"Pages: {result.document.metadata.total_pages}")
print(f"Chunks: {len(result.chunks)}")
print(result.chunks[0].text)
```

## 3. Inspect Chunks

```python
for chunk in doc.chunks[:3]:
    print(f"[{chunk.chunk_type}] tokens={chunk.token_count}")
    print(chunk.text[:200])
    print("---")
```

## 4. Use with LangChain

```python
from longparser.integrations.langchain import LongParserLoader

loader = LongParserLoader("report.pdf")
documents = loader.load()  # Returns List[Document]
```

## 5. Use with LlamaIndex

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

# Run server
uvicorn longparser.server.app:app --reload
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
