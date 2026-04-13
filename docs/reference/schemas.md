# Schemas Reference

Core data models used throughout LongParser.

## Document

Top-level container returned by `DocumentPipeline.process_file()`.

```python
class Document:
    doc_id: str
    metadata: DocumentMetadata
    pages: list[Page]
    blocks: list[Block]
    chunks: list[Chunk]
    extraction_metadata: ExtractionMetadata
```

## Block

A semantic unit extracted from a document (heading, paragraph, table, etc.).

```python
class Block:
    block_id: str
    type: BlockType          # heading | paragraph | table | list_item | equation | ...
    text: str
    order_index: int
    heading_level: int | None
    indent_level: int
    hierarchy_path: list[str]  # ["Introduction", "Methods"]
    provenance: Provenance
    confidence: Confidence
    flags: BlockFlags
    table: Table | None      # populated for table blocks
```

## Chunk

A RAG-ready text chunk with full metadata for retrieval.

```python
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    chunk_type: str          # section | table | table_schema | list | equation
    section_path: list[str]
    page_numbers: list[int]
    block_ids: list[str]     # source block IDs for traceability
    metadata: dict
    equation_detected: bool
```

## BlockType

```python
class BlockType(str, Enum):
    HEADING    = "heading"
    PARAGRAPH  = "paragraph"
    TABLE      = "table"
    LIST_ITEM  = "list_item"
    EQUATION   = "equation"
    FIGURE     = "figure"
    CAPTION    = "caption"
    HEADER     = "header"
    FOOTER     = "footer"
    CODE       = "code"
    FORMULA    = "formula"
```

## ChunkingConfig

```python
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
    detect_equations: bool = True
    exclude_headers_footers: bool = True
    generate_schema_chunks: bool = True
    table_chunk_format: str = "row_record"   # pipe | row_record
    wide_table_col_threshold: int = 15
    min_chunk_tokens: int = 20
```

## Provenance

```python
class Provenance:
    source_file: str
    page_number: int
    bbox: BoundingBox
    extractor: str

class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
```
