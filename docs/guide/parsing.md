# Document Parsing

LongParser uses **Docling** with Tesseract CLI OCR as its extraction engine — supporting PDF, DOCX, PPTX, XLSX, and CSV.

## Supported Formats

| Format | Capabilities |
|---|---|
| **PDF** | Layout analysis, OCR, table structure, equation detection |
| **DOCX** | OMML equations → LaTeX injection |
| **PPTX** | Slide-by-slide extraction with hierarchy |
| **XLSX** | Sheet-aware table chunking with column profiles |
| **CSV** | Column-type inference, schema chunks |

## Basic Usage

```python
from longparser import DocumentPipeline, ProcessingConfig

pipeline = DocumentPipeline(ProcessingConfig())
result = pipeline.process_file("paper.pdf")
```

## Formula Modes

LongParser has three modes for equation handling:

```python
config = ProcessingConfig(formula_mode="smart")
# fast   — Unicode normalization only (fastest)
# smart  — BBox crop → pix2tex OCR for detected formulas
# full   — Docling enrichment enabled for all formulas
```

## Accessing Results

```python
# Pages
for page in result.document.pages:
    print(f"Page {page.page_number}: {page.width}x{page.height}")

# Blocks (semantic units)
for block in result.document.blocks:
    print(f"[{block.type}] p={block.provenance.page_number}: {block.text[:80]}")

# Chunks (RAG-ready)
for chunk in result.chunks:
    print(f"{chunk.chunk_type} | {chunk.token_count} tokens | pages={chunk.page_numbers}")
```

## Block Types

| Type | Description |
|---|---|
| `heading` | Section header (with level) |
| `paragraph` | Body text |
| `table` | Structured table data |
| `list_item` | Bullet or numbered list item |
| `equation` | Mathematical formula |
| `figure` | Image or diagram |
| `caption` | Figure/table caption |
| `header` | Page header |
| `footer` | Page footer |

## RTL Language Support

LongParser automatically detects right-to-left scripts (Arabic, Hebrew, etc.) and applies correct text ordering:

```python
from longparser.utils.rtl_detector import detect_rtl

is_rtl = detect_rtl("مرحبا بالعالم")  # True
```
