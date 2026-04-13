# Pipeline Reference

The `DocumentPipeline` is the main entry point for LongParser's extraction pipeline.

## DocumentPipeline

```python
from longparser import DocumentPipeline, ProcessingConfig

pipeline = DocumentPipeline(ProcessingConfig())
result = pipeline.process_file("document.pdf")
```

### Constructor

```python
DocumentPipeline(config: ProcessingConfig | None = None)
```

| Parameter | Type | Description |
|---|---|---|
| `config` | `ProcessingConfig \| None` | Extraction and chunking configuration (uses defaults if `None`) |

### Methods

#### `process_file(file_path)`

Process a document end-to-end through Extract → Validate → Chunk.

```python
result = pipeline.process_file("report.pdf")
# Returns: longparser.pipeline.PipelineResult
```

**Returns:** `PipelineResult` with `.document` and `.chunks` populated.

#### `process(request)`

Process a document from a `JobRequest` object.

```python
from longparser import JobRequest
request = JobRequest(file_path="report.pdf")
result = pipeline.process(request)
```

#### `process_batch(file_paths)`

Process multiple documents sequentially.

```python
results = pipeline.process_batch(["a.pdf", "b.docx", "c.pptx"])
```

## ProcessingConfig

```python
from longparser import ProcessingConfig

config = ProcessingConfig(
    do_ocr=True,
    do_table_structure=True,
    formula_mode="smart",   # fast | smart | full
    formula_ocr=True,
    export_images=False,
    force_full_page_ocr=False,
    max_pages=None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `do_ocr` | `bool` | `True` | Enable Tesseract OCR |
| `do_table_structure` | `bool` | `True` | Enable TableFormer |
| `formula_mode` | `str` | `"smart"` | Equation parsing mode |
| `formula_ocr` | `bool` | `True` | Enable LaTeX OCR |
| `export_images` | `bool` | `False` | Export figure images |
| `force_full_page_ocr` | `bool` | `False` | OCR entire page |
| `max_pages` | `int \| None` | `None` | Page cap |
