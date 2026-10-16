"""Observable chunking behavior; external model and language resources stay offline."""

from __future__ import annotations

import builtins
import socket
import sys
from types import ModuleType

import pytest

from longparser.chunkers import HybridChunker, quality_scorer, semantic_boundary
from longparser.schemas import BlockType, Chunk, Table, TableCell
from tests.conftest import make_block


@pytest.fixture(autouse=True)
def offline_resources(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Controlled chunker tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(quality_scorer, "_english_words", set())
    monkeypatch.setattr(semantic_boundary, "_models", {})
    language = ModuleType("fast_langdetect")
    language.detect = lambda text: {"score": 1.0}
    monkeypatch.setitem(sys.modules, "fast_langdetect", language)


def test_empty_input(default_config):
    assert HybridChunker(default_config).chunk([]) == []


def test_paragraph_retains_content_and_provenance(default_config, paragraph_block):
    chunks = HybridChunker(default_config).chunk([paragraph_block])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == paragraph_block.text
    assert chunk.chunk_type == "section"
    assert chunk.token_count == 6
    assert chunk.block_ids == [paragraph_block.block_id]
    assert chunk.page_numbers == [1]
    assert chunk.section_path == []


@pytest.mark.parametrize("text", ["", " ", "\n\t", "------"])
def test_blank_input_does_not_fabricate_content(text, small_config):
    assert HybridChunker(small_config).chunk([make_block(text)]) == []


def test_normal_paragraphs_split_at_budget_without_loss(small_config):
    blocks = [make_block("First paragraph " * 10, page=1),
              make_block("Second paragraph " * 10, page=2)]
    chunks = HybridChunker(small_config).chunk(blocks)
    assert [c.text for c in chunks] == [b.text.strip() for b in blocks]
    assert all(c.token_count <= small_config.max_tokens for c in chunks)
    assert [c.block_ids for c in chunks] == [[b.block_id] for b in blocks]
    assert [c.page_numbers for c in chunks] == [[1], [2]]


def test_short_tail_merges_without_losing_sources(small_config):
    config = small_config.model_copy(update={"max_tokens": 27})
    blocks = [make_block("Normal paragraph " * 10), make_block("Short tail.", page=2)]
    chunks = HybridChunker(config).chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].text == blocks[0].text.strip() + "\n\nShort tail."
    assert chunks[0].block_ids == [b.block_id for b in blocks]
    assert chunks[0].page_numbers == [1, 2]
    # min_tokens merging intentionally takes precedence over the packing target.
    assert chunks[0].token_count == 29 > config.max_tokens


def test_extractor_supplied_section_paths(small_config, simple_block_list):
    blocks = [b.model_copy(deep=True) for b in simple_block_list]
    for index, block in enumerate(blocks):
        block.hierarchy_path = ["Introduction"] if index < 2 else ["Introduction", "Background"]
    chunks = HybridChunker(small_config).chunk(blocks)
    assert [c.section_path for c in chunks] == [["Introduction"], ["Introduction", "Background"]]
    assert [c.block_ids for c in chunks] == [[b.block_id for b in blocks[:2]],
                                           [b.block_id for b in blocks[2:]]]
    assert chunks[0].text == blocks[0].text + "\n\n" + blocks[1].text.strip()
    assert chunks[1].text == blocks[2].text + "\n\n" + blocks[3].text.strip()


def test_overlap_copies_previous_context_only_when_enabled(small_config):
    blocks = [make_block("First paragraph " * 10), make_block("Second paragraph " * 10)]
    chunks = HybridChunker(small_config.model_copy(update={"overlap_blocks": 1})).chunk(blocks)
    assert len(chunks) == 2
    assert not chunks[0].overlap_with_previous
    assert chunks[1].overlap_with_previous
    assert chunks[1].text == blocks[0].text.strip() + "\n\n" + blocks[1].text.strip()
    # Source IDs describe the primary chunk; copied context is identified by the overlap flag.
    assert chunks[1].block_ids == [blocks[1].block_id]
    assert chunks[1].token_count > small_config.max_tokens


def test_overlap_does_not_cross_section_boundaries(small_config):
    blocks = [make_block("First section " * 10), make_block("Second section " * 10)]
    for block, path in zip(blocks, [["First"], ["Second"]], strict=True):
        block.hierarchy_path = path
    chunks = HybridChunker(small_config.model_copy(update={"overlap_blocks": 1})).chunk(blocks)
    assert [c.text for c in chunks] == [b.text.strip() for b in blocks]
    assert not any(c.overlap_with_previous for c in chunks)


@pytest.mark.parametrize("text", ["Introduction", "Background"])
def test_heading_only_content_is_retained(text, heading_block, default_config):
    heading = heading_block.model_copy(deep=True, update={"text": text, "hierarchy_path": [text]})
    chunks = HybridChunker(default_config).chunk([heading])
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].section_path == [text]
    assert chunks[0].block_ids == [heading.block_id]


@pytest.fixture
def structured_table():
    block = make_block("Inventory table", BlockType.TABLE, page=3)
    rows = [("Name", "Value"), ("apple", "10"), ("pear", "20"),
            ("plum", "30"), ("kiwi", "40")]
    # Deliberately reverse cell input order to check stable column rendering.
    block.table = Table(n_rows=5, n_cols=2, cells=[
        TableCell(row_index=r, col_index=c, text=row[c])
        for r, row in enumerate(rows) for c in [1, 0]
    ])
    block.hierarchy_path = ["Inventory"]
    return block


def test_structured_table_batches_rows_and_preserves_sources(small_config, structured_table):
    config = small_config.model_copy(update={"max_tokens": 12, "generate_schema_chunks": False})
    chunks = HybridChunker(config).chunk([structured_table])
    assert [c.text for c in chunks] == [
        "Row 1: Name=apple; Value=10\nRow 2: Name=pear; Value=20",
        "Row 3: Name=plum; Value=30\nRow 4: Name=kiwi; Value=40",
    ]
    assert [(c.metadata["row_start"], c.metadata["row_end"]) for c in chunks] == [(1, 2), (3, 4)]
    for chunk in chunks:
        assert chunk.chunk_type == "table"
        assert chunk.block_ids == [structured_table.block_id]
        assert chunk.page_numbers == [3]
        assert chunk.section_path == ["Inventory"]
        assert chunk.token_count <= config.max_tokens


def test_table_only_document_has_schema_and_data(small_config, structured_table):
    config = small_config.model_copy(update={"max_tokens": 512})
    chunks = HybridChunker(config).chunk([structured_table])
    assert [c.chunk_type for c in chunks] == ["table_schema", "table"]
    assert "Name (string, 0% null)" in chunks[0].text
    assert "Value (number, 0% null)" in chunks[0].text
    assert chunks[0].metadata == {"schema": True, "n_rows": 4, "n_cols": 2}
    assert "Name=apple; Value=10" in chunks[1].text
    assert all(c.block_ids == [structured_table.block_id] for c in chunks)


def test_pipe_table_preserves_column_order(small_config, structured_table):
    config = small_config.model_copy(update={"table_chunk_format": "pipe", "generate_schema_chunks": False})
    chunks = HybridChunker(config).chunk([structured_table])
    assert len(chunks) == 1
    assert chunks[0].text == "Name | Value\napple | 10\npear | 20\nplum | 30\nkiwi | 40"


def test_overlap_does_not_copy_table_rows(small_config, structured_table):
    config = small_config.model_copy(update={"max_tokens": 12, "generate_schema_chunks": False,
                                            "overlap_blocks": 1})
    chunks = HybridChunker(config).chunk([structured_table])
    assert len(chunks) == 2
    assert "apple" not in chunks[1].text
    assert not any(c.overlap_with_previous for c in chunks)


def test_unstructured_table_keeps_text(small_config):
    block = make_block("Name | Value\napple | 10", BlockType.TABLE)
    chunks = HybridChunker(small_config).chunk([block])
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
    assert chunks[0].text == block.text


@pytest.mark.parametrize("kind", ["paragraph", "table_row"])
def test_indivisible_oversized_content_is_retained(kind, small_config):
    content = "Detailed description " * 60
    block = make_block(content.strip())
    if kind == "table_row":
        block.type = BlockType.TABLE
        block.table = Table(n_rows=2, n_cols=1, cells=[
            TableCell(row_index=0, col_index=0, text="Description"),
            TableCell(row_index=1, col_index=0, text=content.strip()),
        ])
    config = small_config.model_copy(update={"generate_schema_chunks": False})
    chunks = HybridChunker(config).chunk([block])
    assert len(chunks) == 1
    assert content.strip() in chunks[0].text
    assert chunks[0].token_count > config.max_tokens
    assert chunks[0].block_ids == [block.block_id]


def test_consecutive_headings_preserve_supplied_context(small_config, heading_block):
    first = heading_block.model_copy(deep=True, update={"hierarchy_path": ["Introduction"]})
    second = make_block("Background", BlockType.HEADING, heading_level=2)
    second.hierarchy_path = ["Introduction", "Background"]
    config = small_config.model_copy(update={"min_tokens": 0})
    chunks = HybridChunker(config).chunk([first, second])
    assert [c.text for c in chunks] == ["Introduction", "Background"]
    assert [c.section_path for c in chunks] == [["Introduction"], ["Introduction", "Background"]]


def test_list_group_keeps_lead_in_and_splits_at_bullets(small_config):
    blocks = [make_block("Checklist follows:"),
              make_block("First checklist item " * 7, BlockType.LIST_ITEM),
              make_block("Second checklist item " * 7, BlockType.LIST_ITEM)]
    chunks = HybridChunker(small_config).chunk(blocks)
    assert len(chunks) == 2
    assert all(c.chunk_type == "list" for c in chunks)
    assert chunks[0].text == blocks[0].text + "\n\n" + blocks[1].text.strip()
    assert chunks[1].text == blocks[2].text.strip()
    assert [c.block_ids for c in chunks] == [[b.block_id for b in blocks[:2]], [blocks[2].block_id]]


def test_equation_is_detected_and_kept_with_context(small_config):
    blocks = [make_block("The expression is defined as"), make_block("α + β = γ")]
    chunks = HybridChunker(small_config).chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].text == "The expression is defined as\n\nα + β = γ"
    assert chunks[0].equation_detected
    assert chunks[0].chunk_type == "equation"
    assert chunks[0].block_ids == [b.block_id for b in blocks]


@pytest.mark.parametrize("opening_page,context_page", [(1, 2), (2, 2)])
def test_equation_glue_preserves_cross_page_provenance(small_config, opening_page, context_page):
    blocks = [make_block("Opening paragraph " * 10, page=opening_page),
              make_block("Equation context words " * 5, page=context_page),
              make_block("α + β = γ", BlockType.EQUATION, page=3)]
    chunks = HybridChunker(small_config).chunk(blocks)
    assert [c.block_ids for c in chunks] == [[blocks[0].block_id], [b.block_id for b in blocks[1:]]]
    assert [c.page_numbers for c in chunks] == [[opening_page], [context_page, 3]]


def scored_chunk(text, blocks):
    chunk = Chunk(text=text, token_count=10, chunk_type="section",
                  block_ids=[b.block_id for b in blocks])
    assert quality_scorer.score_chunks([chunk], blocks)[0] is chunk
    return chunk


def test_quality_weights_confidence_by_source_character_length():
    blocks = [make_block("aaaa"), make_block("bbbbbbbb")]
    blocks[0].confidence.overall = 0.2
    blocks[1].confidence.overall = 0.8
    assert scored_chunk("aaaa bbbbbbbb", blocks).quality_score == pytest.approx(0.6)


def test_quality_penalizes_noise_and_clamps_scores():
    clean = make_block("Readable sentence.")
    noisy = make_block("Readable §§§§§§§§§§§§§§§§")
    assert scored_chunk(clean.text, [clean]).quality_score == 1.0
    assert scored_chunk(noisy.text, [noisy]).quality_score < 1.0
    noisy.confidence.overall = 0.1
    assert scored_chunk(noisy.text, [noisy]).quality_score == 0.0


def test_quality_dictionary_penalty(monkeypatch):
    monkeypatch.setattr(quality_scorer, "_english_words", {"readable", "sentence"})
    known = make_block("readable sentence")
    unknown = make_block("zzzz qqqq")
    assert scored_chunk(known.text, [known]).quality_score == 1.0
    assert scored_chunk(unknown.text, [unknown]).quality_score == pytest.approx(0.7)


def test_quality_missing_dictionary_uses_confidence(monkeypatch):
    original_open = builtins.open

    def missing_dictionary(path, *args, **kwargs):
        if str(path) == "/usr/share/dict/words":
            raise FileNotFoundError(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(quality_scorer, "_english_words", None)
    monkeypatch.setattr(builtins, "open", missing_dictionary)
    block = make_block("readable sentence")
    block.confidence.overall = 0.6
    assert scored_chunk(block.text, [block]).quality_score == pytest.approx(0.6)


@pytest.mark.parametrize("outcome", ["low", "error", "short"])
def test_quality_language_detector_is_offline_and_handles_fallback(monkeypatch, outcome):
    calls = []

    def detect(text):
        calls.append(text)
        if outcome == "error":
            raise RuntimeError("Optional detector unavailable")
        return {"score": 0.0}

    monkeypatch.setattr(sys.modules["fast_langdetect"], "detect", detect)
    block = make_block("tiny" if outcome == "short" else "Readable\nsentence")
    score = scored_chunk(block.text, [block]).quality_score
    assert score == pytest.approx(0.6 if outcome == "low" else 1.0)
    assert calls == ([] if outcome == "short" else ["Readable sentence"])


def test_quality_missing_sources_and_empty_inputs():
    block = make_block("Known source")
    chunk = Chunk(text="Unknown source", token_count=2, chunk_type="section", block_ids=["missing"])
    assert quality_scorer.score_chunks([chunk], [block]) == [chunk]
    assert chunk.quality_score == 0.5
    assert quality_scorer.score_chunks([], [block]) == []
    assert quality_scorer.score_chunks([chunk], []) == [chunk]


@pytest.fixture
def fake_embedding_sdk(monkeypatch):
    """Replace only the external SDK; production cosine and packing logic run."""
    import numpy as np

    calls = {"loads": [], "encodes": []}
    vectors = {"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [0.0, 1.0], "Z": [0.0, 0.0]}

    class Model:
        def __init__(self, name):
            calls["loads"].append(name)

        def encode(self, texts, **kwargs):
            calls["encodes"].append((texts, kwargs))
            return np.array([vectors[t] for t in texts])

    sdk = ModuleType("sentence_transformers")
    sdk.SentenceTransformer = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", sdk)
    return calls


@pytest.mark.parametrize("texts", [[], ["A"]])
def test_semantic_short_input_never_loads_model(fake_embedding_sdk, texts):
    assert semantic_boundary.find_semantic_boundaries(texts) == []
    assert fake_embedding_sdk["loads"] == []


@pytest.mark.parametrize("texts,threshold,expected", [
    (["A", "B"], 0.3, []),
    (["A", "B", "C"], 0.3, [2]),
    (["A", "B"], 1.0, []),
    (["A", "C"], 0.0, []),
    (["A", "Z", "B"], 0.3, [1, 2]),
])
def test_semantic_cosine_boundaries_and_strict_threshold(fake_embedding_sdk, texts, threshold, expected):
    assert semantic_boundary.find_semantic_boundaries(texts, threshold) == expected
    assert fake_embedding_sdk["encodes"] == [(texts, {"batch_size": 64, "show_progress_bar": False})]


def test_semantic_model_is_reused_for_same_name(fake_embedding_sdk):
    for _ in range(2):
        assert semantic_boundary.find_semantic_boundaries(["A", "C"], model_name="test-model") == [1]
    assert fake_embedding_sdk["loads"] == ["test-model"]


def test_semantic_missing_optional_dependency_is_graceful(monkeypatch):
    original_import = builtins.__import__

    def missing_sdk(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("SDK deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sdk)
    assert semantic_boundary.find_semantic_boundaries(["A", "C"]) == []


def test_semantic_disabled_never_loads_model(small_config, fake_embedding_sdk):
    chunks = HybridChunker(small_config).chunk([make_block("A"), make_block("C")])
    assert len(chunks) == 1
    assert fake_embedding_sdk["loads"] == []
    assert fake_embedding_sdk["encodes"] == []


def test_semantic_enabled_splits_before_later_block_with_blank_filtering(small_config, fake_embedding_sdk):
    blocks = [make_block("A"), make_block("   "), make_block("B"), make_block("C")]
    config = small_config.model_copy(update={"use_semantic_chunking": True, "min_tokens": 0,
                                            "detect_equations": False})
    chunks = HybridChunker(config).chunk(blocks)
    assert [c.text for c in chunks] == ["A\n\nB", "C"]
    assert [c.block_ids for c in chunks] == [[blocks[0].block_id, blocks[2].block_id], [blocks[3].block_id]]
    assert fake_embedding_sdk["encodes"][0][0] == ["A", "B", "C"]


def test_table_captions_preserve_text_and_primary_references(small_config, structured_table):
    before = make_block("Inventory explanation", BlockType.CAPTION, page=3)
    after = make_block("Values measured today", BlockType.CAPTION, page=3)
    before.hierarchy_path = after.hierarchy_path = ["Inventory"]
    config = small_config.model_copy(update={"max_tokens": 512, "generate_schema_chunks": False})
    chunks = HybridChunker(config).chunk([before, structured_table, after])
    assert len(chunks) == 1
    assert chunks[0].text.startswith(before.text + "\n\nRow 1:")
    assert chunks[0].text.endswith("\n\n" + after.text)
    assert chunks[0].block_ids == [before.block_id, structured_table.block_id, after.block_id]
    assert chunks[0].page_numbers == [3]


def test_header_footer_filter_can_be_disabled(small_config):
    blocks = [make_block("Document header", BlockType.HEADER), make_block("Body text"),
              make_block("Document footer", BlockType.FOOTER)]
    filtered = HybridChunker(small_config).chunk([b.model_copy(deep=True) for b in blocks])
    assert [c.text for c in filtered] == ["Body text"]
    retained = HybridChunker(small_config.model_copy(update={"exclude_headers_footers": False})).chunk(blocks)
    assert [c.text for c in retained] == ["Document header\n\nBody text\n\nDocument footer"]


@pytest.mark.parametrize("sections,counts,equations,expected", [
    (["A", "A"], [1, 20], [True, False], 1),
    (["A", "B"], [1, 20], [True, False], 1),
    (["A", "B"], [20, 1], [False, True], 1),
    (["A"], [1], [True], 1),
    (["A", "A"], [20, 1], [False, True], 1),
])
def test_small_chunk_merging_preserves_content_sources_pages_and_equations(sections, counts, equations, expected):
    from longparser.schemas import ChunkingConfig
    chunks = [Chunk(text=f"part-{i}", token_count=count, chunk_type="section", section_path=[section],
                    block_ids=[f"block-{i}"], page_numbers=[i], equation_detected=equation)
              for i, (section, count, equation) in enumerate(zip(sections, counts, equations, strict=True))]
    merged = HybridChunker(ChunkingConfig(min_tokens=10))._merge_small_chunks(chunks)
    assert len(merged) == expected
    assert merged[0].text == "\n\n".join(f"part-{i}" for i in range(len(chunks)))
    assert merged[0].block_ids == [f"block-{i}" for i in range(len(chunks))]
    assert merged[0].page_numbers == list(range(len(chunks)))
    assert merged[0].equation_detected
    if len(chunks) > 1:
        assert merged[0].chunk_type == "equation"


def test_overlap_does_not_duplicate_equation_regions():
    from longparser.schemas import ChunkingConfig
    chunks = [Chunk(text="⟦EQUATION⟧\nx+y\n⟦/EQUATION⟧", token_count=20, chunk_type="equation"),
              Chunk(text="explanation", token_count=1, chunk_type="section")]
    HybridChunker(ChunkingConfig(overlap_blocks=1))._apply_overlap(chunks)
    assert chunks[1].text == "explanation" and not chunks[1].overlap_with_previous


@pytest.mark.parametrize("table_format", ["row_record", "pipe"])
def test_wide_table_banding_repeats_key_and_preserves_every_value(table_format):
    from longparser.schemas import ChunkingConfig
    cells = [TableCell(r0=row, c0=column, text=f"column-{column}" if row == 0 else f"value-{column}")
             for row in range(2) for column in range(30)]
    block = make_block("Wide table", BlockType.TABLE)
    block.table = Table(n_rows=2, n_cols=30, cells=cells)
    chunks = HybridChunker(ChunkingConfig(min_tokens=0, max_tokens=1000, wide_table_col_threshold=25,
                                        generate_schema_chunks=False, table_chunk_format=table_format)).chunk([block])
    assert len(chunks) == 3
    assert all("value-0" in chunk.text for chunk in chunks)
    assert all(f"value-{column}" in "\n".join(chunk.text for chunk in chunks) for column in range(30))
    assert all(chunk.block_ids == [block.block_id] for chunk in chunks)
