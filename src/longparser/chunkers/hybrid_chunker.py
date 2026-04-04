"""
Hybrid Chunker for LongParser — RAG-optimized document splitting.

Combines 6 strategies:
  0. Autonomous equation detection (pre-pass)
  1. Structure-aware (hierarchical) chunking
  2. Layout-aware block classification
  3. Token-window packing with overlap
  4. Table-aware chunking
  5. List-aware chunking
"""

from __future__ import annotations

import re
import logging
import unicodedata
from typing import Optional

from ..schemas import Block, BlockType, Chunk, ChunkingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for autonomous equation detection
# ---------------------------------------------------------------------------

# Greek letters (lowercase + uppercase)
_GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")

# Common math operators & symbols
_MATH_SYMBOLS = set("∑∏∫∂∇±×÷≤≥≠≈∞∈∉⊂⊃⊆⊇∪∩∧∨¬⊕⊗→←↔⇒⇐⇔∀∃∅⟨⟩⟦⟧")

# Subscript / superscript Unicode ranges
_SUB_SUPER = set("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ"
                 "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")

# Combined set for fast lookup
_MATH_CHARS = _GREEK | _MATH_SYMBOLS | _SUB_SUPER

# Regex patterns for equation-like text
_EQ_PATTERNS = [
    # Variable subscript patterns: x_i, y_j, a_1, etc.
    re.compile(r"\b[a-zA-Z]_[a-zA-Z0-9]+\b"),
    # Inline math with equals: f(x) = ..., y = ...
    re.compile(r"[a-zA-Z]\s*\([a-zA-Z,\s]*\)\s*="),
    # Summation/product patterns
    re.compile(r"∑|∏|∫|Σ|Π"),
    # Fractions and common math notation
    re.compile(r"\b(?:frac|sqrt|log|exp|sin|cos|tan|argmax|argmin)\b"),
    # LaTeX-like patterns that may survive OCR
    re.compile(r"\\[a-zA-Z]+"),
    # Tensor/dimension-like notation: d1, d2, d3, n(D)
    re.compile(r"\b[a-z]\d+\b"),
    # Comma-separated Greek or single-letter variables: κ,λ,ν or x, y, z
    re.compile(r"[α-ωΑ-Ω],\s*[α-ωΑ-Ω]"),
    # Parenthetical math notation: ( ) D n, (x), f(x)
    re.compile(r"\(\s*\)\s*[A-Z]\s*[a-z]"),
    # Scattered single-letter math variables with spaces: n y n y, a i y
    re.compile(r"(?:\b[a-z]\b\s+){3,}"),
    # Cardinality / dimensionality notation
    re.compile(r"\b(?:cardinality|dimension|tensor|kernel|initializer)\b", re.IGNORECASE),
]

# Phrases that introduce or surround equations
_EQ_LEAD_PHRASES = re.compile(
    r"(?:defined\s+as|given\s+by|expressed\s+as|computed\s+as|"
    r"formally|where\s+\w+\s+(?:is|are|denotes?))",
    re.IGNORECASE,
)

# Phrases at the end of a block that introduce an upcoming equation
_EQ_TAIL_PHRASES = re.compile(
    r"(?:defined\s+as\s*[,.]?\s*$|given\s+by\s*[,.]?\s*$|"
    r"expressed\s+as\s*[,.]?\s*$|computed\s+as\s*[,.]?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Tokenizer (word-split approximation, no tiktoken dependency)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count (≈ 0.75 tokens per whitespace-split word)."""
    words = text.split()
    return max(1, int(len(words) * 1.33))  # words * 1.33 ≈ tokens


# ---------------------------------------------------------------------------
# Strategy 0: Autonomous equation detection
# ---------------------------------------------------------------------------

# Regex for blocks that contain only separator characters
_SEPARATOR_ONLY = re.compile(r"^[\s_\-=\.·•─━═]+$")


def _is_separator_only(text: str) -> bool:
    """Return True if text is only separator chars (underscores, dashes, etc.)."""
    return bool(text and _SEPARATOR_ONLY.match(text.strip()))


def _math_char_density(text: str) -> float:
    """Fraction of chars that are math-class Unicode."""
    if not text:
        return 0.0
    count = sum(1 for ch in text if ch in _MATH_CHARS or
                unicodedata.category(ch) in ("Sm", "So"))
    return count / len(text)


def _eq_pattern_hits(text: str) -> int:
    """Count how many equation regex patterns match."""
    return sum(1 for pat in _EQ_PATTERNS if pat.search(text))


def _is_equation_candidate(block: Block, prev_block: Optional[Block] = None) -> bool:
    """
    Determine if a paragraph block should be re-tagged as an equation.

    Scoring heuristics (threshold = 2.0):
      - Math-char density > 5% → +2.0;  > 1% → +1.0
      - Equation pattern hits ≥ 3 → +2.0;  ≥ 2 → +1.5;  ≥ 1 → +0.5
      - Short block (< 80 chars) with isolated variables → +1.0
      - Previous block ends with lead-in phrase → +1.5
      - Self text starts with or contains lead-in phrase → +1.0
      - Greek letters ≥ 2 → +1.5;  ≥ 1 → +0.5
    """
    if block.type != BlockType.PARAGRAPH:
        return False

    text = block.text.strip()
    if not text:
        return False

    # Math-char density
    density = _math_char_density(text)

    # Pattern hits
    hits = _eq_pattern_hits(text)

    # Short block (< 80 chars) with variable-like single letters
    is_short = len(text) < 80
    has_isolated_vars = bool(re.search(r"(?<!\w)[a-zA-Z](?!\w)", text))

    # Previous block leads into equation
    has_lead_in = False
    if prev_block and prev_block.text:
        prev_tail = prev_block.text.strip()[-120:]
        if _EQ_LEAD_PHRASES.search(prev_tail) or _EQ_TAIL_PHRASES.search(prev_tail):
            has_lead_in = True

    # Self text contains equation-contextual phrases
    has_self_context = bool(_EQ_LEAD_PHRASES.search(text))

    # --- Scoring ---
    score = 0.0

    # Density scoring (lowered thresholds for OCR'd math text)
    if density > 0.05:
        score += 2.0
    elif density > 0.01:
        score += 1.0

    # Pattern hits
    if hits >= 3:
        score += 2.0
    elif hits >= 2:
        score += 1.5
    elif hits >= 1:
        score += 0.5

    # Short math fragments
    if is_short and has_isolated_vars:
        score += 1.0

    # Contextual cues
    if has_lead_in:
        score += 1.5

    if has_self_context:
        score += 1.0

    # Greek letter presence
    greek_count = sum(1 for ch in text if ch in _GREEK)
    if greek_count >= 2:
        score += 1.5
    elif greek_count >= 1:
        score += 0.5

    return score >= 2.0


def _detect_equations(blocks: list[Block]) -> list[Block]:
    """
    Pre-pass: re-tag paragraph blocks that look like equations.
    Returns the modified block list (original list is mutated in place).
    """
    retagged = 0
    for i, block in enumerate(blocks):
        prev = blocks[i - 1] if i > 0 else None
        if _is_equation_candidate(block, prev):
            logger.info(
                f"  [EQ-DETECT] Re-tagged block {block.block_id} "
                f"(order={block.order_index}, page={block.provenance.page_number}) "
                f"as equation — preview: {block.text[:80]!r}"
            )
            block.type = BlockType.EQUATION
            retagged += 1

    logger.info(f"  [EQ-DETECT] Re-tagged {retagged} paragraph(s) → equation")
    return blocks


# ---------------------------------------------------------------------------
# Strategy 4: Table-aware chunking (with smart rendering + profiling)
# ---------------------------------------------------------------------------


def _build_ordered_grid(table) -> dict[int, dict[int, str]]:
    """
    Build a 2D dict from table cells: rows[r][c] = text.
    Enforces column order (Fix B).
    """
    rows: dict[int, dict[int, str]] = {}
    for cell in table.cells:
        r = cell.row_index
        c = cell.col_index
        rows.setdefault(r, {})[c] = cell.text
    return rows


def _detect_header_rows(table) -> list[int]:
    """
    Detect header row indices using column_header flag (Gap #2).
    Falls back to row 0 if no flags are set.
    """
    header_rows = set()
    for cell in table.cells:
        # Check if the cell has a column_header flag (from Docling)
        if getattr(cell, 'column_header', False):
            header_rows.add(cell.row_index)
    
    if header_rows:
        return sorted(header_rows)
    # Fallback: treat row 0 as header
    return [0]


def _get_column_names(grid: dict[int, dict[int, str]], header_rows: list[int], n_cols: int) -> list[str]:
    """
    Extract column names from header rows.
    Synthesizes col_0..col_n if headers are empty.
    """
    names = [""] * n_cols
    for hr in header_rows:
        row_data = grid.get(hr, {})
        for c in range(n_cols):
            val = row_data.get(c, "").strip()
            if val:
                if names[c]:
                    names[c] += f" {val}"
                else:
                    names[c] = val
    
    # Synthesize if still empty
    for c in range(n_cols):
        if not names[c]:
            names[c] = f"col_{c}"
    
    return names


def _render_row_as_record(row_idx: int, row_data: dict[int, str], col_names: list[str], n_cols: int) -> str:
    """Render a single row as: Row N: col_a=val; col_b=val; ..."""
    parts = []
    for c in range(n_cols):
        val = row_data.get(c, "").strip()
        if val:
            parts.append(f"{col_names[c]}={val}")
    return f"Row {row_idx}: " + "; ".join(parts) if parts else ""


def _render_row_as_pipe(row_data: dict[int, str], n_cols: int) -> str:
    """Render a single row as pipe-delimited."""
    return " | ".join(row_data.get(c, "") for c in range(n_cols))


def _guess_col_type(values: list[str]) -> str:
    """Guess column type from sample values."""
    if not values:
        return "string"
    
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "string"
    
    # Check numeric
    num_count = 0
    for v in non_empty[:20]:  # Sample first 20
        try:
            float(v.replace(",", "").replace("$", "").replace("%", ""))
            num_count += 1
        except ValueError:
            pass
    if num_count > len(non_empty[:20]) * 0.7:
        return "number"
    
    # Check date-like
    import re
    date_pattern = re.compile(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}')
    date_count = sum(1 for v in non_empty[:20] if date_pattern.search(v))
    if date_count > len(non_empty[:20]) * 0.5:
        return "date"
    
    return "string"


def _generate_schema_chunk(
    block, table, grid, header_rows, col_names, n_cols, data_row_indices
) -> Chunk:
    """
    Generate a schema chunk for a table (Fix E + Gap #5).
    Contains: table info, column list with types, null rates, sample rows.
    """
    page = block.provenance.page_number
    n_data = len(data_row_indices)
    
    # Column profiling
    col_profiles = []
    for c in range(n_cols):
        values = [grid.get(r, {}).get(c, "") for r in data_row_indices]
        col_type = _guess_col_type(values)
        total = len(values)
        null_count = sum(1 for v in values if not v.strip())
        null_pct = f"{(null_count / total * 100):.0f}%" if total > 0 else "0%"
        col_profiles.append(f"  - {col_names[c]} ({col_type}, {null_pct} null)")
    
    # Sample rows (first 3–5)
    sample_count = min(5, n_data)
    sample_rows = []
    for i, r_idx in enumerate(data_row_indices[:sample_count]):
        row_data = grid.get(r_idx, {})
        parts = [f"{col_names[c]}={row_data.get(c, '')}" for c in range(n_cols)]
        sample_rows.append(f"  Row {r_idx}: " + "; ".join(parts))
    
    lines = [
        f"[TABLE SCHEMA]",
        f"Table ID: {block.block_id}",
        f"Rows: {n_data} (data rows), Columns: {n_cols}",
        f"Columns:",
    ]
    lines.extend(col_profiles)
    lines.append(f"Sample Rows ({sample_count}):")
    lines.extend(sample_rows)
    
    schema_text = "\n".join(lines)
    return Chunk(
        text=schema_text,
        token_count=_count_tokens(schema_text),
        chunk_type="table_schema",
        section_path=list(block.hierarchy_path),
        page_numbers=[page],
        block_ids=[block.block_id],
        metadata={"schema": True, "n_rows": n_data, "n_cols": n_cols},
    )


def _chunk_table(block: Block, config: ChunkingConfig) -> list[Chunk]:
    """
    Create chunks from a table block.
    
    Implements:
      Fix B: Column-ordered rendering
      Fix C: Token-aware row batching (header repeated)
      Fix D: Row-as-record format for RAG
      Fix E: Schema chunk per table
      Fix F: Wide-table column banding
      Gap #2: Smart header detection
      Gap #4: Chunk metadata (row ranges)
      Gap #5: Schema chunk profiling
    """
    chunks: list[Chunk] = []
    table = block.table
    page = block.provenance.page_number

    if not table:
        # No structured table data — fallback to single text chunk
        chunks.append(Chunk(
            text=block.text,
            token_count=_count_tokens(block.text),
            chunk_type="table",
            section_path=list(block.hierarchy_path),
            page_numbers=[page],
            block_ids=[block.block_id],
        ))
        return chunks

    n_rows = table.n_rows
    n_cols = table.n_cols
    
    # Fix B: Build ordered grid
    grid = _build_ordered_grid(table)
    
    # Gap #2: Detect header rows
    header_rows = _detect_header_rows(table)
    col_names = _get_column_names(grid, header_rows, n_cols)
    
    # Data rows = all rows not in header
    header_set = set(header_rows)
    data_row_indices = sorted(r for r in grid.keys() if r not in header_set)
    
    # Fix E + Gap #5: Schema chunk
    if config.generate_schema_chunks and data_row_indices:
        schema_chunk = _generate_schema_chunk(
            block, table, grid, header_rows, col_names, n_cols, data_row_indices
        )
        chunks.append(schema_chunk)
    
    # Fix F: Wide-table column banding
    if n_cols > config.wide_table_col_threshold:
        # Keep col 0 as key column, split remaining into bands
        key_col = 0
        remaining_cols = list(range(1, n_cols))
        band_size = 12
        bands = []
        for i in range(0, len(remaining_cols), band_size):
            band_cols = [key_col] + remaining_cols[i:i + band_size]
            bands.append(band_cols)
    else:
        bands = [list(range(n_cols))]  # Single band with all columns
    
    # Process each band
    for band_idx, band_cols in enumerate(bands):
        band_col_names = [col_names[c] for c in band_cols]
        
        # Build header text for pipe format
        if config.table_chunk_format == "pipe":
            header_text = " | ".join(band_col_names)
        else:
            header_text = ""  # Not needed for row_record; names are inline
        
        # Fix C: Token-aware row batching
        current_row_texts: list[str] = []
        current_tokens = _count_tokens(header_text) if header_text else 0
        chunk_row_start = data_row_indices[0] if data_row_indices else 0
        
        for r_idx in data_row_indices:
            row_data = {c: grid.get(r_idx, {}).get(c, "") for c in band_cols}
            
            # Fix D: Render based on format
            if config.table_chunk_format == "row_record":
                row_text = _render_row_as_record(r_idx, row_data, band_col_names, len(band_cols))
            else:
                row_text = _render_row_as_pipe(row_data, len(band_cols))
            
            if not row_text.strip():
                continue
            
            row_tokens = _count_tokens(row_text)
            
            # Would adding this row exceed budget?
            if current_tokens + row_tokens > config.max_tokens and current_row_texts:
                # Flush current chunk
                if config.table_chunk_format == "pipe" and header_text:
                    chunk_text = header_text + "\n" + "\n".join(current_row_texts)
                else:
                    chunk_text = "\n".join(current_row_texts)
                
                chunk_row_end = r_idx - 1
                meta = {
                    "row_start": chunk_row_start,
                    "row_end": chunk_row_end,
                    "col_band": band_cols if len(bands) > 1 else None,
                }
                
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=_count_tokens(chunk_text),
                    chunk_type="table",
                    section_path=list(block.hierarchy_path),
                    page_numbers=[page],
                    block_ids=[block.block_id],
                    metadata=meta,
                ))
                
                current_row_texts = []
                current_tokens = _count_tokens(header_text) if header_text else 0
                chunk_row_start = r_idx
            
            current_row_texts.append(row_text)
            current_tokens += row_tokens
        
        # Flush remaining rows
        if current_row_texts:
            if config.table_chunk_format == "pipe" and header_text:
                chunk_text = header_text + "\n" + "\n".join(current_row_texts)
            else:
                chunk_text = "\n".join(current_row_texts)
            
            chunk_row_end = data_row_indices[-1] if data_row_indices else chunk_row_start
            meta = {
                "row_start": chunk_row_start,
                "row_end": chunk_row_end,
                "col_band": band_cols if len(bands) > 1 else None,
            }
            
            chunks.append(Chunk(
                text=chunk_text,
                token_count=_count_tokens(chunk_text),
                chunk_type="table",
                section_path=list(block.hierarchy_path),
                page_numbers=[page],
                block_ids=[block.block_id],
                metadata=meta,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 5: List-aware chunking
# ---------------------------------------------------------------------------

def _extract_list_groups(blocks: list[Block]) -> list[tuple[int, int]]:
    """
    Identify contiguous list_item sequences with their lead-in paragraph.
    Returns list of (start_index, end_index) inclusive ranges.
    """
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(blocks):
        if blocks[i].type == BlockType.LIST_ITEM:
            # Look back for a lead-in paragraph
            start = i
            if i > 0 and blocks[i - 1].type == BlockType.PARAGRAPH:
                start = i - 1

            # Extend to all consecutive list items
            end = i
            while end + 1 < len(blocks) and blocks[end + 1].type == BlockType.LIST_ITEM:
                end += 1

            groups.append((start, end))
            i = end + 1
        else:
            i += 1

    return groups


# ---------------------------------------------------------------------------
# Main chunker class
# ---------------------------------------------------------------------------

class HybridChunker:
    """
    Hybrid chunking engine combining 6 strategies for RAG-optimized output.

    Usage:
        chunker = HybridChunker(ChunkingConfig())
        chunks = chunker.chunk(blocks)
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        """
        Run the full hybrid chunking pipeline on a list of blocks.

        Steps:
          0. Autonomous equation detection (re-tag missed equations)
          1. Filter header/footer blocks
          2. Group by section (hierarchy_path)
          3. Per section: table-aware → list-aware → token-window packing
          4. Apply overlap between consecutive chunks within a section
        """
        logger.info(f"[HybridChunker] Starting — {len(blocks)} blocks, "
                    f"max_tokens={self.config.max_tokens}")

        # --- Strategy 0: equation detection ---
        if self.config.detect_equations:
            blocks = _detect_equations(blocks)

        # --- Filter headers/footers ---
        if self.config.exclude_headers_footers:
            before = len(blocks)
            blocks = [
                b for b in blocks
                if b.type not in (BlockType.HEADER, BlockType.FOOTER)
            ]
            filtered = before - len(blocks)
            if filtered:
                logger.info(f"  Filtered {filtered} header/footer block(s)")

        # --- Filter separator-only blocks (underscores, dashes, etc.) ---
        before = len(blocks)
        blocks = [
            b for b in blocks
            if not _is_separator_only(b.text)
        ]
        sep_filtered = before - len(blocks)
        if sep_filtered:
            logger.info(f"  Filtered {sep_filtered} separator-only block(s)")

        # --- Strategy 1: group by section ---
        section_groups = self._group_by_section(blocks)
        logger.info(f"  {len(section_groups)} section group(s)")

        all_chunks: list[Chunk] = []

        for section_path, section_blocks in section_groups:
            section_chunks = self._chunk_section(section_path, section_blocks)
            all_chunks.extend(section_chunks)

        # --- Merge small chunks ---
        all_chunks = self._merge_small_chunks(all_chunks)

        # --- Apply overlap ---
        all_chunks = self._apply_overlap(all_chunks)

        logger.info(f"[HybridChunker] Done — {len(all_chunks)} chunks produced")
        return all_chunks

    # -----------------------------------------------------------------------
    # Strategy 1: Structure-aware grouping
    # -----------------------------------------------------------------------

    def _group_by_section(
        self, blocks: list[Block]
    ) -> list[tuple[list[str], list[Block]]]:
        """Group blocks by hierarchy_path (section boundaries)."""
        groups: list[tuple[list[str], list[Block]]] = []
        current_path: list[str] | None = None
        current_blocks: list[Block] = []

        for block in blocks:
            path = tuple(block.hierarchy_path)
            if path != tuple(current_path or []):
                if current_blocks:
                    groups.append((list(current_path or []), current_blocks))
                current_path = list(block.hierarchy_path)
                current_blocks = [block]
            else:
                current_blocks.append(block)

        if current_blocks:
            groups.append((list(current_path or []), current_blocks))

        return groups

    # -----------------------------------------------------------------------
    # Per-section chunking
    # -----------------------------------------------------------------------

    def _chunk_section(
        self, section_path: list[str], blocks: list[Block]
    ) -> list[Chunk]:
        """
        Process a section's blocks using strategies 2-5.

        Order:
          - Tables → dedicated table chunks
          - List groups → dedicated list chunks
          - Equations → kept with surrounding context
          - Remaining → token-window packing
        """
        chunks: list[Chunk] = []

        # Identify which blocks are consumed by tables or list groups
        consumed: set[int] = set()

        # --- Strategy 4: Table-aware ---
        for i, block in enumerate(blocks):
            if block.type == BlockType.TABLE:
                table_chunks = _chunk_table(block, self.config)
                chunks.extend(table_chunks)
                consumed.add(i)
                # Also consume adjacent captions
                if i > 0 and blocks[i - 1].type == BlockType.CAPTION:
                    # Prepend caption to first table chunk
                    if table_chunks and blocks[i - 1].text:
                        table_chunks[0].text = blocks[i - 1].text + "\n\n" + table_chunks[0].text
                        table_chunks[0].token_count = _count_tokens(table_chunks[0].text)
                        table_chunks[0].block_ids.insert(0, blocks[i - 1].block_id)
                    consumed.add(i - 1)
                if i + 1 < len(blocks) and blocks[i + 1].type == BlockType.CAPTION:
                    if table_chunks:
                        table_chunks[-1].text += "\n\n" + blocks[i + 1].text
                        table_chunks[-1].token_count = _count_tokens(table_chunks[-1].text)
                        table_chunks[-1].block_ids.append(blocks[i + 1].block_id)
                    consumed.add(i + 1)

        # --- Strategy 5: List-aware ---
        remaining = [(i, b) for i, b in enumerate(blocks) if i not in consumed]

        # Find list groups in the remaining blocks
        remaining_blocks = [b for _, b in remaining]
        remaining_indices = [i for i, _ in remaining]
        list_groups = _extract_list_groups(remaining_blocks)

        list_consumed_local: set[int] = set()  # indices into remaining_blocks
        for group_start, group_end in list_groups:
            group_blocks = remaining_blocks[group_start:group_end + 1]
            group_text = "\n\n".join(b.text for b in group_blocks if b.text)
            tokens = _count_tokens(group_text)

            if tokens <= self.config.max_tokens:
                chunks.append(Chunk(
                    text=group_text,
                    token_count=tokens,
                    chunk_type="list",
                    section_path=section_path,
                    page_numbers=sorted(set(
                        b.provenance.page_number for b in group_blocks
                    )),
                    block_ids=[b.block_id for b in group_blocks],
                ))
            else:
                # Split list at bullet boundaries
                chunks.extend(
                    self._split_list_group(group_blocks, section_path)
                )

            for j in range(group_start, group_end + 1):
                list_consumed_local.add(j)
                consumed.add(remaining_indices[j])

        # --- Strategy 3: Token-window packing for remaining blocks ---
        final_remaining = [
            b for i, b in enumerate(blocks) if i not in consumed
        ]
        if final_remaining:
            packed = self._pack_blocks(final_remaining, section_path)
            chunks.extend(packed)

        return chunks

    # -----------------------------------------------------------------------
    # Strategy 3: Token-window packing
    # -----------------------------------------------------------------------

    def _pack_blocks(
        self, blocks: list[Block], section_path: list[str]
    ) -> list[Chunk]:
        """
        Pack blocks into chunks respecting max_tokens.
        Equations are kept with their surrounding context.
        """
        chunks: list[Chunk] = []
        current_texts: list[str] = []
        current_ids: list[str] = []
        current_pages: set[int] = set()
        current_tokens = 0
        has_equation = False

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            block_tokens = _count_tokens(text)

            # If adding this block would exceed the limit, flush
            if (current_tokens + block_tokens > self.config.max_tokens
                    and current_texts):
                
                carry_text = None
                carry_id = None
                carry_tokens = 0
                
                # Glue logic: if next block (this block) is an equation, 
                # keep the last paragraph with it
                if block.type == BlockType.EQUATION and len(current_texts) > 0:
                    carry_text = current_texts.pop()
                    carry_id = current_ids.pop()
                    carry_tokens = _count_tokens(carry_text)
                    current_tokens -= carry_tokens
                
                chunk_text = "\n\n".join(current_texts)
                if chunk_text.strip():
                    chunk_type = "equation" if has_equation else "section"
                    chunks.append(Chunk(
                        text=chunk_text,
                        token_count=_count_tokens(chunk_text),
                        chunk_type=chunk_type,
                        section_path=section_path,
                        page_numbers=sorted(current_pages),
                        block_ids=list(current_ids),
                        equation_detected=has_equation,
                    ))
                
                current_texts = []
                current_ids = []
                current_pages = set()
                current_tokens = 0
                has_equation = False
                
                if carry_text:
                    current_texts.append(carry_text)
                    current_ids.append(carry_id)
                    current_tokens += carry_tokens
                    # Note: We assume the carried block is close enough to the next block 
                    # that simply adding the next block's page will suffice for provenance.

            current_texts.append(text)
            current_ids.append(block.block_id)
            current_pages.add(block.provenance.page_number)
            current_tokens += block_tokens

            if block.type == BlockType.EQUATION:
                has_equation = True

        # Flush remaining
        if current_texts:
            chunk_text = "\n\n".join(current_texts)
            chunk_type = "equation" if has_equation else "section"
            # Only emit if meets min_tokens or is the only content
            if _count_tokens(chunk_text) >= self.config.min_tokens or not chunks:
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=_count_tokens(chunk_text),
                    chunk_type=chunk_type,
                    section_path=section_path,
                    page_numbers=sorted(current_pages),
                    block_ids=list(current_ids),
                    equation_detected=has_equation,
                ))
            elif chunks:
                # Merge into previous chunk
                prev = chunks[-1]
                prev.text += "\n\n" + chunk_text
                prev.token_count = _count_tokens(prev.text)
                prev.block_ids.extend(current_ids)
                prev.page_numbers = sorted(
                    set(prev.page_numbers) | current_pages
                )
                if has_equation:
                    prev.equation_detected = True
                    prev.chunk_type = "equation"

        return chunks

    # -----------------------------------------------------------------------
    # List splitting helper
    # -----------------------------------------------------------------------

    def _split_list_group(
        self, blocks: list[Block], section_path: list[str]
    ) -> list[Chunk]:
        """Split a list group that exceeds max_tokens at bullet boundaries."""
        chunks: list[Chunk] = []
        current_texts: list[str] = []
        current_ids: list[str] = []
        current_pages: set[int] = set()
        current_tokens = 0

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            block_tokens = _count_tokens(text)

            if (current_tokens + block_tokens > self.config.max_tokens
                    and current_texts):
                chunk_text = "\n\n".join(current_texts)
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=_count_tokens(chunk_text),
                    chunk_type="list",
                    section_path=section_path,
                    page_numbers=sorted(current_pages),
                    block_ids=list(current_ids),
                ))
                current_texts = []
                current_ids = []
                current_pages = set()
                current_tokens = 0

            current_texts.append(text)
            current_ids.append(block.block_id)
            current_pages.add(block.provenance.page_number)
            current_tokens += block_tokens

        if current_texts:
            chunk_text = "\n\n".join(current_texts)
            chunks.append(Chunk(
                text=chunk_text,
                token_count=_count_tokens(chunk_text),
                chunk_type="list",
                section_path=section_path,
                page_numbers=sorted(current_pages),
                block_ids=list(current_ids),
            ))

        return chunks

    # -----------------------------------------------------------------------
    # Merge small chunks
    # -----------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Post-processing pass: merge any chunk below min_tokens into its
        nearest neighbor within the same section.  Preference order:
          1. Previous chunk (same section_path)
          2. Next chunk (same section_path)
          3. Previous chunk (different section — avoid data loss)
        """
        if not chunks or self.config.min_tokens <= 0:
            return chunks

        merged: list[Chunk] = []

        for chunk in chunks:
            # If large enough, keep
            if chunk.token_count >= self.config.min_tokens:
                merged.append(chunk)
                continue

            # --- Try to merge into previous chunk (same section) ---
            if merged and merged[-1].section_path == chunk.section_path:
                self._absorb(merged[-1], chunk)
                logger.debug(
                    f"  [MERGE] Merged small chunk {chunk.token_count} "
                    f"tokens into previous (same section)"
                )
                continue

            # --- No previous same-section neighbor; buffer it ---
            merged.append(chunk)

        # Second pass: merge any remaining small chunks forward
        final: list[Chunk] = []
        for i, chunk in enumerate(merged):
            if chunk.token_count < self.config.min_tokens:
                # 1. Try next chunk (same section)
                if i + 1 < len(merged) and merged[i + 1].section_path == chunk.section_path:
                    self._absorb_prepend(merged[i + 1], chunk)
                    logger.debug(
                        f"  [MERGE] Merged small chunk {chunk.token_count} "
                        f"tokens into next (same section)"
                    )
                    continue

                # 2. Try previous chunk (any section, fallback)
                if final:
                    self._absorb(final[-1], chunk)
                    logger.debug(
                        f"  [MERGE] Merged small chunk {chunk.token_count} "
                        f"tokens into previous (cross-section)"
                    )
                    continue

                # 3. Try next chunk (any section, fallback - e.g. first chunk)
                if i + 1 < len(merged):
                    self._absorb_prepend(merged[i + 1], chunk)
                    logger.debug(
                        f"  [MERGE] Merged small chunk {chunk.token_count} "
                        f"tokens into next (cross-section)"
                    )
                    continue
                
                # 4. Total fallback: if isolated and TINY, ignore it? Or keep?
                # For now, we keep it if we can't merge anywhere.
                logger.warning(
                    f"  [MERGE] Could not merge isolated small chunk: {chunk.token_count} tokens"
                )

            final.append(chunk)



        before = len(chunks)
        after = len(final)
        if before != after:
            logger.info(
                f"  [MERGE] Merged {before - after} small chunk(s) "
                f"(min_tokens={self.config.min_tokens}): {before} → {after}"
            )

        return final

    @staticmethod
    def _absorb(target: Chunk, small: Chunk) -> None:
        """Append small chunk content into target."""
        target.text += "\n\n" + small.text
        target.token_count = _count_tokens(target.text)
        target.block_ids.extend(small.block_ids)
        target.page_numbers = sorted(set(target.page_numbers) | set(small.page_numbers))
        if small.equation_detected:
            target.equation_detected = True
            target.chunk_type = "equation"

    @staticmethod
    def _absorb_prepend(target: Chunk, small: Chunk) -> None:
        """Prepend small chunk content into target."""
        target.text = small.text + "\n\n" + target.text
        target.token_count = _count_tokens(target.text)
        target.block_ids = small.block_ids + target.block_ids
        target.page_numbers = sorted(set(target.page_numbers) | set(small.page_numbers))
        if small.equation_detected:
            target.equation_detected = True
            target.chunk_type = "equation"

    # -----------------------------------------------------------------------
    # Overlap
    # -----------------------------------------------------------------------

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Apply block-level overlap between consecutive chunks
        within the same section.
        """
        if self.config.overlap_blocks <= 0 or len(chunks) < 2:
            return chunks

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            # Only overlap within same section
            if prev.section_path != curr.section_path:
                continue

            # Skip overlap for table chunks
            if prev.chunk_type == "table" or curr.chunk_type == "table":
                continue

            # Get last N paragraphs of previous chunk as overlap
            prev_parts = prev.text.split("\n\n")
            overlap_parts = prev_parts[-self.config.overlap_blocks:]

            # Avoid duplicating equations in overlap
            if any("⟦EQUATION⟧" in part for part in overlap_parts):
                continue


            if overlap_parts:
                overlap_text = "\n\n".join(overlap_parts)
                curr.text = overlap_text + "\n\n" + curr.text
                curr.token_count = _count_tokens(curr.text)
                curr.overlap_with_previous = True

        return chunks
