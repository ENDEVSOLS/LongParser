# Hybrid Chunking

LongParser's `HybridChunker` combines **6 strategies** to produce RAG-optimized chunks that respect document structure.

## The 6 Strategies

| # | Strategy | What it does |
|---|---|---|
| 0 | **Equation detection** | Pre-pass re-tags missed formula blocks |
| 1 | **Structure-aware** | Groups blocks by hierarchy path (sections) |
| 2 | **Layout-aware** | Classifies blocks by type before chunking |
| 3 | **Token-window packing** | Packs blocks up to `max_tokens` with overlap |
| 4 | **Table-aware** | Schema chunk + row-batched table chunks |
| 5 | **List-aware** | Keeps bullet groups together as single chunks |

## Usage

```python
from longparser.chunkers import HybridChunker
from longparser.schemas import ChunkingConfig

config = ChunkingConfig(
    max_tokens=512,
    overlap_tokens=64,
    detect_equations=True,
    table_chunk_format="row_record",  # or "pipe"
    generate_schema_chunks=True,
    use_semantic_chunking=True,       # Split on semantic topic shifts
)

chunker = HybridChunker(config)
chunks = chunker.chunk(doc.blocks)
```

## Table Chunking

Tables are chunked with:
- **Schema chunk** — column names, types, null rates, sample rows
- **Data chunks** — row-batched with header repeated per chunk
- **Wide-table banding** — columns split into bands for very wide tables

```python
# Table chunk formats
ChunkingConfig(table_chunk_format="row_record")
# Row 1: col_a=val; col_b=val; col_c=val

ChunkingConfig(table_chunk_format="pipe")
# col_a | col_b | col_c
# val_1 | val_2 | val_3
```

## Chunk Structure

```python
@dataclass
class Chunk:
    chunk_id: str           # UUID
    text: str               # Chunk content
    token_count: int        # Approximate token count
    chunk_type: str         # section | table | table_schema | list | equation
    section_path: list[str] # ["Introduction", "Methods"]
    page_numbers: list[int] # [1, 2]
    block_ids: list[str]    # Source block IDs for traceability
    metadata: dict          # chunk_type-specific metadata
```

## Token Budget

Chunks respect a hard `max_tokens` ceiling. Equations are kept with their surrounding context using a **glue heuristic**:

- If the *next* block is an equation AND the current window overflows, the last paragraph carries over into the new chunk (so the equation is never split from its context).

## Semantic Chunking

When `use_semantic_chunking=True`, the chunker uses embedding similarity (default: `all-MiniLM-L6-v2`) to detect topic shifts within a section. Instead of splitting purely by token count, it finds natural breakpoints where the semantic content changes.

```python
config = ChunkingConfig(
    use_semantic_chunking=True,
    semantic_threshold=0.3,              # Lower = more splits
    semantic_model="all-MiniLM-L6-v2",   # or "all-mpnet-base-v2"
)
```

The model is lazily loaded on first use — no memory cost if the feature is disabled.

## Cross-Reference Resolution

When `resolve_cross_references=True` (default), the pipeline automatically links textual references to their target blocks:

- **Explicit references:** `"see Figure 3"`, `"Table 2"`, `"Section 3.1"`, `"Appendix A"` → linked via regex + dictionary lookup.
- **Implicit references:** `"the figure above"`, `"the table below"` → linked via spatial proximity in reading order.

Resolved links appear in chunk metadata:

```json
{
  "cross_references": [
    {"label": "Figure 3", "target_block_id": "block-uuid-123"},
    {"label": "the table below", "target_block_id": "block-uuid-456", "resolution": "proximity"}
  ]
}
```

## Quality Scoring

Each chunk receives a `quality_score` (0.0–1.0) based on:

- **Block confidence** — OCR confidence from the extraction engine
- **Dictionary word coverage** — percentage of words found in `/usr/share/dict/words` (penalizes garbled OCR)
- **Language ID confidence** — fastText-based language detection score (low confidence = noise)

```python
from longparser.chunkers.quality_scorer import score_chunks
scored = score_chunks(chunks, blocks)
print(scored[0].quality_score)  # 0.92
```

## PII Redaction

When `redact_pii=True` in `ProcessingConfig`, the pipeline automatically masks sensitive data **before** any HITL review:

- **Pass 1 (always):** Fast regex + Luhn checksum for Emails, Phones, SSNs, Credit Cards, IPs.
- **Pass 2 (optional):** spaCy NER (`use_ner_redaction=True`) for names, organizations, and locations.

Original values are preserved in `block.pii_redactions` for authorized recovery.
