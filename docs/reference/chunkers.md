# Chunkers Reference

## HybridChunker

The main chunking engine combining 6 strategies for RAG-optimized output.

```python
from longparser.chunkers import HybridChunker
from longparser.schemas import ChunkingConfig

chunker = HybridChunker(config=ChunkingConfig())
chunks = chunker.chunk(blocks)
```

### Constructor

```python
HybridChunker(config: ChunkingConfig | None = None)
```

If `config` is `None`, uses default `ChunkingConfig()`.

### Methods

#### `chunk(blocks)`

Run the full 6-strategy pipeline on a list of `Block` objects.

```python
chunks: list[Chunk] = chunker.chunk(doc.blocks)
```

**Steps:**

1. **Strategy 0** — Autonomous equation detection pre-pass
2. **Filter** — Remove header/footer blocks and separator-only blocks
3. **Strategy 1** — Group blocks by `hierarchy_path` (sections)
4. **Per section:**
   - **Strategy 4** — Table-aware chunking (schema + data chunks)
   - **Strategy 5** — List-aware chunking (group bullet lists)
   - **Strategy 3** — Token-window packing for remaining blocks
5. **Merge** small chunks below `min_chunk_tokens`
6. **Overlap** — Add token overlap between consecutive chunks

### Configuration Reference

| Config | Default | Description |
|---|---|---|
| `max_tokens` | `512` | Hard ceiling per chunk |
| `overlap_tokens` | `64` | Overlap between chunks |
| `detect_equations` | `True` | Run equation detection pre-pass |
| `exclude_headers_footers` | `True` | Remove page headers/footers |
| `generate_schema_chunks` | `True` | Add schema chunk per table |
| `table_chunk_format` | `"row_record"` | `pipe` or `row_record` |
| `wide_table_col_threshold` | `15` | Split columns into bands above this |
| `min_chunk_tokens` | `20` | Merge chunks smaller than this |
