"""Offline pipeline and privacy regression tests at the agreed public seams."""

from __future__ import annotations

import builtins
import logging
import socket
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import pytest

from longparser.chunkers import quality_scorer
from longparser.schemas import BlockType, Document, DocumentMetadata, Page, ProcessingConfig
from tests.conftest import make_block
from tests.pipeline_doubles import install_docling_sdk_stubs


def document_with(*blocks):
    return Document(metadata=DocumentMetadata(source_file="synthetic.txt", total_pages=1),
                    pages=[Page(page_number=1, width=100, height=100, blocks=list(blocks))])


@pytest.fixture(autouse=True)
def offline_resources(monkeypatch):
    install_docling_sdk_stubs(monkeypatch)
    def deny_network(*args, **kwargs):
        raise AssertionError("Milestone 4 tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(quality_scorer, "_english_words", set())
    detector = ModuleType("fast_langdetect")
    detector.detect = lambda text: {"lang": "en", "score": 1.0}
    monkeypatch.setitem(sys.modules, "fast_langdetect", detector)


@pytest.mark.parametrize("text,category,placeholder", [
    ("sample@example.test", "emails", "[REDACTED_EMAIL_1]"),
    ("202-555-0100", "phones", "[REDACTED_PHONE_1]"),
    ("202.555.0100", "phones", "[REDACTED_PHONE_1]"),
    ("202 555 0100", "phones", "[REDACTED_PHONE_1]"),
    ("+442055501234", "phones", "[REDACTED_PHONE_1]"),
    ("000-12-3456", "ssns", "[REDACTED_SSN_1]"),
    ("4111 1111 1111 1111", "credit_cards", "[REDACTED_CC_1]"),
    ("192.0.2.10", "ip_addresses", "[REDACTED_IP_1]"),
])
def test_pii_patterns_replace_visible_text_and_preserve_original(text, category, placeholder, caplog):
    from longparser.pipeline.pii_redactor import redact_document

    block = make_block("Value: " + text)
    document = document_with(block)
    with caplog.at_level(logging.DEBUG):
        result, report = redact_document(document)
    assert result is document
    assert block.text == "Value: " + placeholder
    assert block.pii_redactions == {placeholder: text}
    assert asdict(report) == {name: int(name == category) for name in asdict(report)}
    assert text not in report.summary()
    assert text not in caplog.text


def test_invalid_card_is_not_counted_as_a_credit_card():
    from longparser.pipeline.pii_redactor import redact_document

    block = make_block("Invalid checksum: 4111 1111 1111 1112")
    _, report = redact_document(document_with(block))
    assert report.credit_cards == 0
    assert not any(key.startswith("[REDACTED_CC_") for key in block.pii_redactions)
    # A different pattern may still match part of this value; do not equate that with card validation.


@pytest.mark.parametrize("text", ["999.0.2.10", "256.0.2.10", "Ordinary prose.", "", " \n\t"])
def test_clean_blank_and_invalid_ip_have_no_redactions(text):
    from longparser.pipeline.pii_redactor import redact_document

    block = make_block(text)
    _, report = redact_document(document_with(block))
    assert block.text == text
    assert block.pii_redactions == {}
    assert all(value == 0 for value in asdict(report).values())
    assert report.summary() == "No PII found"


def test_mixed_pii_counts_match_visible_placeholders_and_original_mapping():
    from longparser.pipeline.pii_redactor import redact_document

    original = "sample@example.test; 202-555-0100; 000-12-3456; 4111 1111 1111 1111; 192.0.2.10"
    block = make_block(original)
    _, report = redact_document(document_with(block))
    expected = {"[REDACTED_EMAIL_1]": "sample@example.test", "[REDACTED_PHONE_1]": "202-555-0100",
                "[REDACTED_SSN_1]": "000-12-3456", "[REDACTED_CC_1]": "4111 1111 1111 1111",
                "[REDACTED_IP_1]": "192.0.2.10"}
    assert block.pii_redactions == expected
    assert block.text == "[REDACTED_EMAIL_1]; [REDACTED_PHONE_1]; [REDACTED_SSN_1]; [REDACTED_CC_1]; [REDACTED_IP_1]"
    assert asdict(report) == dict(emails=1, phones=1, ssns=1, credit_cards=1,
                                 ip_addresses=1, names=0, organizations=0, locations=0)


@pytest.mark.parametrize("failure", ["package_absent", "model_absent"])
def test_regex_redaction_survives_missing_ner_without_download(monkeypatch, failure, caplog):
    from longparser.pipeline import pii_redactor

    monkeypatch.setattr(pii_redactor, "_nlp_models", {})
    if failure == "package_absent":
        original_import = builtins.__import__

        def unavailable(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("spaCy deliberately absent")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", unavailable)
    else:
        spacy = ModuleType("spacy")

        def missing_model(name):
            assert name == "synthetic_model"
            raise OSError("Model unavailable locally")

        spacy.load = missing_model
        monkeypatch.setitem(sys.modules, "spacy", spacy)
    block = make_block("sample@example.test")
    _, report = pii_redactor.redact_document(document_with(block), use_ner=True, ner_model="synthetic_model")
    assert report.emails == 1
    assert block.text == "[REDACTED_EMAIL_1]"
    assert block.pii_redactions == {"[REDACTED_EMAIL_1]": "sample@example.test"}
    assert "sample@example.test" not in caplog.text


@pytest.fixture
def pipeline_env(monkeypatch):
    from longparser.pipeline import orchestrator

    calls = []
    source = document_with(make_block("Extracted body text."))
    source.all_blocks[0].hierarchy_path = ["Introduction"]
    hierarchy = [SimpleNamespace(text="Extracted body text.", heading_path=["Introduction"],
                                 level=1, page_number=1)]

    class FakeDocling:
        def __init__(self, **kwargs):
            self.options = kwargs
            self._languages = ["eng"]
            calls.append(("construct", "docling"))

        def extract(self, path, config):
            calls.append(("extract", "docling", path, config))
            return source, SimpleNamespace(strategy_used="standard")

        def get_hierarchy(self, path, config):
            calls.append(("hierarchy", path))
            return hierarchy

    class FakeNative:
        def __init__(self):
            calls.append(("construct", "pymupdf"))

        def extract(self, path, config):
            calls.append(("extract", "pymupdf", path, config))
            return source, SimpleNamespace(strategy_used="native")

    class FakeMarker(FakeNative):
        def __init__(self):
            calls.append(("construct", "marker"))

        def extract(self, path, config):
            calls.append(("extract", "marker", path, config))
            return source, SimpleNamespace(strategy_used="marker")

    monkeypatch.setattr(orchestrator, "DoclingExtractor", FakeDocling)
    for name, symbol, extractor in [
        ("pymupdf_extractor", "PyMuPDFExtractor", FakeNative),
        ("marker_extractor", "MarkerExtractor", FakeMarker),
    ]:
        module = ModuleType("longparser.extractors." + name)
        setattr(module, symbol, extractor)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(module=orchestrator, calls=calls, document=source, hierarchy=hierarchy)


@pytest.mark.parametrize("backend", ["docling", "pymupdf", "marker", "auto", "unknown"])
def test_pipeline_selects_requested_backend_and_preserves_unknown_fallback(pipeline_env, backend):
    env = pipeline_env
    pipeline = env.module.PipelineOrchestrator(ProcessingConfig(backend=backend))
    expected = "docling" if backend in ("auto", "unknown") else backend
    assert env.calls == [("construct", expected)]
    assert pipeline._backend_name == ("docling" if backend == "unknown" else backend)


def test_pipeline_default_and_public_aliases(pipeline_env):
    from longparser import DocumentPipeline as root_alias
    from longparser.pipeline import DocumentPipeline, PipelineOrchestrator

    assert root_alias is DocumentPipeline is PipelineOrchestrator
    pipeline = PipelineOrchestrator()
    assert pipeline._backend_name == "docling"
    assert pipeline_env.calls == [("construct", "docling")]


@pytest.mark.parametrize("suffix,sample,expected", [
    (".pdf", "native text " * 12, "pymupdf"),
    (".pdf", "scanned", "docling"),
    (".pdf", "x" * 100, "docling"),
    (".pdf", "x" * 101, "pymupdf"),
    (".docx", "native text " * 12, "docling"),
])
def test_auto_routes_per_document(pipeline_env, monkeypatch, tmp_path, suffix, sample, expected):
    env = pipeline_env
    monkeypatch.setattr(env.module, "extract_sample_text", lambda *args, **kwargs: sample)
    pipeline = env.module.PipelineOrchestrator(ProcessingConfig(backend="auto"))
    result = pipeline.process_file(tmp_path / ("synthetic" + suffix), ProcessingConfig(auto_detect_language=False))
    assert [call[1] for call in env.calls if call[0] == "extract"] == [expected]
    assert result.document is env.document
    assert result.hierarchy == (env.hierarchy if expected == "docling" else [])


def test_auto_missing_actual_pymupdf_sdk_falls_back_to_docling(pipeline_env, monkeypatch, tmp_path):
    env = pipeline_env
    # Import the real backend module and run its real constructor guard.
    monkeypatch.delitem(sys.modules, "longparser.extractors.pymupdf_extractor")
    original_import = builtins.__import__
    attempted = []

    def missing_sdk(name, *args, **kwargs):
        if name == "pymupdf4llm":
            attempted.append(name)
            raise ImportError("Optional SDK deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sdk)
    monkeypatch.setattr(env.module, "extract_sample_text", lambda *args, **kwargs: "native text " * 12)
    pipeline = env.module.PipelineOrchestrator(ProcessingConfig(backend="auto"))
    result = pipeline.process_file(
        tmp_path / "synthetic.pdf", ProcessingConfig(auto_detect_language=False, languages=["ara"])
    )
    assert pipeline.extractor._languages == ["ara"]
    assert attempted == ["pymupdf4llm"]
    assert result.document is env.document
    assert [call[1] for call in env.calls if call[0] == "extract"] == ["docling"]


def test_mocked_extraction_reaches_real_chunking_with_sources(pipeline_env, tmp_path):
    from longparser.schemas import ChunkingConfig

    env = pipeline_env
    pipeline = env.module.PipelineOrchestrator()
    config = ProcessingConfig(auto_detect_language=False, languages=["ara"])
    result = pipeline.process_file(tmp_path / "synthetic.txt", config)
    chunks = pipeline.chunk(result, ChunkingConfig(min_tokens=0, overlap_blocks=0))
    assert result.chunks is chunks
    assert result.total_blocks == 1
    assert result.hierarchy == env.hierarchy
    assert [c.text for c in chunks] == ["Extracted body text."]
    assert chunks[0].section_path == ["Introduction"]
    assert chunks[0].block_ids == [env.document.all_blocks[0].block_id]
    assert chunks[0].page_numbers == [1]
    assert pipeline.extractor._languages == ["ara"]
    assert next(call for call in env.calls if call[0] == "extract")[3] is config


def test_pipeline_redaction_default_is_disabled(pipeline_env, tmp_path):
    assert ProcessingConfig().redact_pii is False
    block = pipeline_env.document.all_blocks[0]
    block.text = "sample@example.test"
    pipeline = pipeline_env.module.PipelineOrchestrator()
    result = pipeline.process_file(tmp_path / "synthetic.txt", ProcessingConfig(auto_detect_language=False))
    assert result.document.all_blocks[0].text == "sample@example.test"
    assert block.pii_redactions == {}


def test_pipeline_redacts_before_chunking_and_report_does_not_log_values(pipeline_env, tmp_path, caplog):
    from longparser.schemas import ChunkingConfig

    block = pipeline_env.document.all_blocks[0]
    block.text = "Contact sample@example.test"
    pipeline = pipeline_env.module.PipelineOrchestrator()
    with caplog.at_level(logging.INFO):
        result = pipeline.process_file(tmp_path / "synthetic.txt",
                                       ProcessingConfig(redact_pii=True, auto_detect_language=False))
        chunks = pipeline.chunk(result, ChunkingConfig(min_tokens=0, overlap_blocks=0))
    assert chunks[0].text == "Contact [REDACTED_EMAIL_1]"
    assert block.pii_redactions == {"[REDACTED_EMAIL_1]": "sample@example.test"}
    assert "PII Redaction Report: 1 emails" in caplog.text
    assert "sample@example.test" not in caplog.text


def test_ner_redacts_supported_entities_and_preserves_originals(monkeypatch):
    from longparser.pipeline import pii_redactor

    entities = [("Alex Example", "PERSON"), ("Example Labs", "ORG"),
                ("Sample City", "GPE"), ("Sample Park", "LOC")]
    text = "Alex Example works at Example Labs in Sample City near Sample Park."
    calls = []

    def nlp(value):
        return SimpleNamespace(ents=[SimpleNamespace(text=name, label_=label,
            start_char=value.index(name), end_char=value.index(name) + len(name))
            for name, label in entities])

    sdk = ModuleType("spacy")
    sdk.load = lambda model: (calls.append(model) or nlp)
    monkeypatch.setitem(sys.modules, "spacy", sdk)
    monkeypatch.setattr(pii_redactor, "_nlp_models", {})
    block = make_block(text)
    _, report = pii_redactor.redact_document(document_with(block), use_ner=True)
    assert block.text == "[REDACTED_NAME_1] works at [REDACTED_ORG_1] in [REDACTED_LOC_2] near [REDACTED_LOC_1]."
    assert block.pii_redactions == {"[REDACTED_NAME_1]": "Alex Example",
                                   "[REDACTED_ORG_1]": "Example Labs",
                                   "[REDACTED_LOC_2]": "Sample City",
                                   "[REDACTED_LOC_1]": "Sample Park"}
    assert (report.names, report.organizations, report.locations) == (1, 1, 2)
    pii_redactor.redact_document(document_with(make_block(text)), use_ner=True)
    assert calls == ["en_core_web_sm"]


def source_chunk(block, text=None):
    from longparser.schemas import Chunk

    return Chunk(text=block.text if text is None else text, token_count=10, chunk_type="section",
                 block_ids=[block.block_id], page_numbers=[block.provenance.page_number])


def test_explicit_reference_resolves_once_and_does_not_invent_targets():
    from longparser.pipeline.cross_reference import resolve_cross_references

    anchor = make_block("See Figure 3 and Figure 3, but Figure 99 is missing.")
    target = make_block("Figure 3: Test chart", BlockType.FIGURE)
    chunks = [source_chunk(anchor)]
    result = resolve_cross_references(document_with(anchor, target), chunks)
    assert result is chunks
    assert chunks[0].metadata["cross_references"] == [{"label": "Figure 3", "target_block_id": target.block_id}]


@pytest.mark.parametrize("kind,phrase,direction", [
    (BlockType.TABLE, "the table below", "after"),
    (BlockType.TABLE, "the table above", "before"),
    (BlockType.FIGURE, "the figure following", "after"),
    (BlockType.FIGURE, "the figure previous", "before"),
])
def test_implicit_reference_uses_nearest_document_order(kind, phrase, direction):
    from longparser.pipeline.cross_reference import resolve_cross_references

    far_before = make_block("Far before", kind, order_index=100)
    near_before = make_block("Near before", kind, order_index=99)
    anchor = make_block("Consult " + phrase, order_index=0)
    near_after = make_block("Near after", kind, order_index=-1)
    far_after = make_block("Far after", kind, order_index=-2)
    document = document_with(far_before, near_before, anchor, near_after, far_after)
    chunk = source_chunk(anchor)
    resolve_cross_references(document, [chunk])
    expected = near_before if direction == "before" else near_after
    assert chunk.metadata["cross_references"] == [{"label": phrase, "target_block_id": expected.block_id,
                                                   "resolution": "proximity"}]


@pytest.mark.parametrize("text,ids", [("No references.", "known"), ("See Figure 99.", "known"),
                                     ("the table below", "unknown"), ("the table above", "known")])
def test_unresolved_reference_does_not_create_links(text, ids):
    from longparser.pipeline.cross_reference import resolve_cross_references

    anchor = make_block(text)
    table = make_block("Unnumbered table", BlockType.TABLE)
    chunk = source_chunk(anchor)
    if ids == "unknown":
        chunk.block_ids = ["missing"]
    resolve_cross_references(document_with(anchor, table), [chunk])
    assert "cross_references" not in chunk.metadata


def test_empty_reference_inputs_are_stable():
    from longparser.pipeline.cross_reference import resolve_cross_references

    assert resolve_cross_references(document_with(), []) == []
    chunk = source_chunk(make_block("No references"))
    assert resolve_cross_references(None, [chunk]) == [chunk]


@pytest.mark.parametrize("target_count", [8, 16])
def test_explicit_reference_index_reads_each_target_label_once(target_count):
    from longparser.pipeline.cross_reference import resolve_cross_references

    reads = []

    class Target:
        type = BlockType.FIGURE

        def __init__(self, number):
            self.number = number
            self.block_id = f"target-{number}"

        @property
        def text(self):
            reads.append(self.number)
            return f"Figure {self.number}: Synthetic caption"

    targets = [Target(number) for number in range(1, target_count + 1)]
    anchor = make_block("See Figure 1")
    chunks = [source_chunk(anchor) for _ in range(32)]
    resolve_cross_references(SimpleNamespace(all_blocks=targets + [anchor]), chunks)
    assert reads == list(range(1, target_count + 1))
    assert all(chunk.metadata["cross_references"] == [{"label": "Figure 1", "target_block_id": "target-1"}]
               for chunk in chunks)
    # This guards explicit target-label indexing, not the complexity of implicit lookup.


@pytest.fixture
def summary_env(monkeypatch):
    import asyncio

    from longparser.pipeline.summary_enricher import generate_summary_chunks

    state = SimpleNamespace(rows=[], prompts=[], factory_calls=[], db_calls=[], failing_title=None,
                            active=0, peak=0, tokenizer_failure=False)

    class Tokenizer:
        def encode(self, text):
            return list(text)

        def decode(self, tokens):
            return "".join(tokens)

    tokenizer = ModuleType("tiktoken")

    def encoding(name):
        assert name == "cl100k_base"
        if state.tokenizer_failure:
            raise OSError("Tokenizer asset unavailable locally")
        return Tokenizer()

    tokenizer.get_encoding = encoding
    monkeypatch.setitem(sys.modules, "tiktoken", tokenizer)
    messages = ModuleType("langchain_core.messages")
    messages.HumanMessage = messages.SystemMessage = SimpleNamespace
    core = ModuleType("langchain_core")
    core.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_core", core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)

    class LLM:
        async def ainvoke(self, prompt):
            state.prompts.append(prompt)
            title = prompt[1].content.split("\n", 1)[0].removeprefix("Section: ")
            state.active += 1
            state.peak = max(state.peak, state.active)
            try:
                await asyncio.sleep(0)  # Yield to other tasks; no timing-based assertion.
                if title == state.failing_title:
                    raise RuntimeError("Synthetic provider failure")
                return SimpleNamespace(content=f"  Summary of {title}.  ")
            finally:
                state.active -= 1

    factory = ModuleType("longparser.server.chat.llm_chain")

    def get_model(**kwargs):
        state.factory_calls.append(kwargs)
        return LLM()

    factory.get_plain_chat_model = get_model
    monkeypatch.setitem(sys.modules, factory.__name__, factory)

    class DB:
        async def get_chunks(self, tenant_id, job_id):
            state.db_calls.append((tenant_id, job_id))
            return state.rows

    state.run = lambda **kwargs: generate_summary_chunks(DB(), "test-tenant", "test-job", **kwargs)
    return state


@pytest.mark.asyncio
async def test_summaries_group_sections_and_skip_irrelevant_chunks(summary_env):
    env = summary_env
    env.rows = [dict(chunk_type="section", section_path=["Intro"], text="First source."),
                dict(chunk_type="equation", section_path=["Intro"], text="Second source."),
                dict(chunk_type="section", section_path=["Methods"], text="Method source."),
                dict(chunk_type="summary", section_path=["Intro"], text="Do not summarize again."),
                dict(chunk_type="figure", section_path=["Images"], text="Image"),
                dict(chunk_type="table", section_path=["Data"], text="Table")]
    chunks = await env.run(provider="synthetic", model="test-model")
    by_path = {tuple(chunk["section_path"]): chunk for chunk in chunks}
    assert set(by_path) == {("Intro",), ("Methods",)}
    assert by_path[("Intro",)]["text"] == "Summary of Intro."
    assert by_path[("Methods",)]["text"] == "Summary of Methods."
    for chunk in chunks:
        assert chunk["chunk_type"] == "summary"
        assert chunk["metadata"] == {"generated_by": "synthetic/test-model"}
        assert chunk["page_numbers"] == chunk["block_ids"] == []
        assert chunk["quality_score"] == 1.0
    prompts = {prompt[1].content.split("\n", 1)[0]: prompt[1].content for prompt in env.prompts}
    assert prompts["Section: Intro"] == "Section: Intro\n---\nFirst source.\n\nSecond source."
    assert prompts["Section: Methods"] == "Section: Methods\n---\nMethod source."
    assert env.db_calls == [("test-tenant", "test-job")]
    assert env.factory_calls == [{"provider": "synthetic", "model": "test-model"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [dict(chunk_type="table", text="Ignored")],
                                     [dict(chunk_type="summary", text="Existing")]])
async def test_empty_or_irrelevant_summary_input_makes_no_model_request(summary_env, rows):
    summary_env.rows = rows
    assert await summary_env.run() == []
    assert summary_env.prompts == []


@pytest.mark.asyncio
async def test_summary_failure_keeps_other_section_results(summary_env, caplog):
    env = summary_env
    env.rows = [dict(chunk_type="section", section_path=["Failure"], text="Failing source text."),
                dict(chunk_type="section", section_path=[], text="Root source.")]
    env.failing_title = "Failure"
    chunks = await env.run()
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Summary of Document Start."
    assert chunks[0]["section_path"] == []
    assert chunks[0]["metadata"] == {"generated_by": "gemini/default"}
    assert "Synthetic provider failure" in caplog.text
    assert "Failing source text." not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback,expected_length", [(False, 2003), (True, 8003)])
async def test_summary_truncation_uses_controlled_tokenizer_or_fallback(summary_env, fallback, expected_length):
    env = summary_env
    env.tokenizer_failure = fallback
    env.rows = [dict(chunk_type="section", section_path=[], text="x" * 9000)]
    assert len(await env.run()) == 1
    source = env.prompts[0][1].content.split("\n---\n", 1)[1]
    assert source == "x" * (expected_length - 3) + "..."


@pytest.mark.asyncio
async def test_summary_requests_respect_concurrency_limit(summary_env):
    env = summary_env
    env.rows = [dict(chunk_type="section", section_path=[title], text="Source text.")
                for title in ["One", "Two", "Three", "Four"]]
    chunks = await env.run(max_concurrent=2)
    assert env.peak == 2
    assert {tuple(chunk["section_path"]) for chunk in chunks} == {("One",), ("Two",), ("Three",), ("Four",)}


@pytest.mark.parametrize("score,expected", [(0.99, ("ar", 0.99)), (0.5, ("ar", 0.5)),
                                          (0.49, ("en", 0.49))])
def test_language_detection_and_confidence_boundary(monkeypatch, score, expected):
    from longparser.utils.lang_detect import detect_language

    calls = []
    monkeypatch.setattr(sys.modules["fast_langdetect"], "detect",
                        lambda text: (calls.append(text) or {"lang": "ar", "score": score}))
    text = "Synthetic language sample with enough characters."
    assert detect_language(text) == expected
    assert calls == [text]


@pytest.mark.parametrize("failure", ["absent", "error"])
def test_language_dependency_failure_defaults_to_english(monkeypatch, failure):
    from longparser.utils.lang_detect import detect_language

    if failure == "absent":
        original_import = builtins.__import__

        def absent(name, *args, **kwargs):
            if name == "fast_langdetect":
                raise ImportError("Detector deliberately absent")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", absent)
    else:
        def error(text):
            raise RuntimeError("Synthetic detector failure")

        monkeypatch.setattr(sys.modules["fast_langdetect"], "detect", error)
    assert detect_language("Synthetic sample with enough characters.") == ("en", 0.0)


@pytest.mark.parametrize("text", ["", "short", " " * 25])
def test_short_language_input_never_calls_detector(monkeypatch, text):
    from longparser.utils.lang_detect import detect_language

    def unexpected(text):
        raise AssertionError("Short input must not invoke the detector")

    monkeypatch.setattr(sys.modules["fast_langdetect"], "detect", unexpected)
    assert detect_language(text) == ("en", 0.0)


@pytest.mark.parametrize("code,expected", [("ar", ["ara"]), ("ur", ["urd"]),
                                         ("zh", ["chi_sim", "chi_tra"]), ("unknown", ["eng"])])
def test_language_code_mapping(code, expected):
    from longparser.utils.lang_detect import get_tesseract_langs

    assert get_tesseract_langs(code) == expected


@pytest.mark.parametrize("explicit,base,auto,expected", [
    (["fra"], ["deu"], True, ["fra"]),
    (None, ["deu"], False, ["deu"]),
    (None, None, False, ["eng"]),
    (None, ["deu"], True, ["ara"]),
])
def test_pipeline_language_priority_and_detected_metadata(pipeline_env, monkeypatch, tmp_path,
                                                         explicit, base, auto, expected):
    env = pipeline_env
    monkeypatch.setattr(env.module, "extract_sample_text", lambda *args, **kwargs: "Synthetic sample " * 4)
    monkeypatch.setattr(sys.modules["fast_langdetect"], "detect", lambda text: {"lang": "ar", "score": 0.95})
    pipeline = env.module.PipelineOrchestrator(tesseract_lang=base)
    result = pipeline.process_file(tmp_path / "synthetic.txt",
                                   ProcessingConfig(languages=explicit, auto_detect_language=auto))
    assert pipeline.extractor._languages == expected
    if expected == ["ara"]:
        assert result.document.metadata.detected_language == "ar"
        assert result.document.metadata.language_confidence == 0.95
    else:
        assert result.document.metadata.detected_language is None


def test_text_sample_reader_and_unsupported_or_missing_files(tmp_path):
    from longparser.utils.lang_detect import extract_sample_text

    text = tmp_path / "sample.txt"
    text.write_text("Synthetic sample", encoding="utf-8")
    assert extract_sample_text(text, max_chars=9) == "Synthetic"
    unsupported = tmp_path / "sample.docx"
    unsupported.write_bytes(b"Unparsed format")
    assert extract_sample_text(unsupported) == ""
    assert extract_sample_text(tmp_path / "missing.txt") == ""


def test_text_sample_reader_failure_is_graceful(monkeypatch, tmp_path):
    from longparser.utils.lang_detect import extract_sample_text

    path = tmp_path / "sample.txt"
    path.write_text("Synthetic text")
    original_open = builtins.open

    def failing(file, *args, **kwargs):
        if str(file) == str(path):
            raise PermissionError("Synthetic read failure")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing)
    assert extract_sample_text(path) == ""


def test_pdf_sample_reader_uses_external_reader_and_truncates(monkeypatch, tmp_path):
    from longparser.utils.lang_detect import extract_sample_text

    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"Synthetic PDF stub")
    sdk = ModuleType("pdfplumber")

    class PDF:
        pages = [SimpleNamespace(extract_text=lambda: "abc"), SimpleNamespace(extract_text=lambda: "def")]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    sdk.open = lambda filename: PDF()
    monkeypatch.setitem(sys.modules, "pdfplumber", sdk)
    assert extract_sample_text(path, max_chars=5) == "abc\nd"


@pytest.mark.parametrize("failure", ["absent", "error"])
def test_pdf_sample_reader_falls_back_to_printable_bytes(monkeypatch, tmp_path, failure):
    from longparser.utils.lang_detect import extract_sample_text

    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"alpha\x00beta\n")
    if failure == "absent":
        original_import = builtins.__import__

        def absent(name, *args, **kwargs):
            if name == "pdfplumber":
                raise ImportError("Reader deliberately absent")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", absent)
    else:
        sdk = ModuleType("pdfplumber")

        def failing(path):
            raise RuntimeError("Synthetic PDF reader failure")

        sdk.open = failing
        monkeypatch.setitem(sys.modules, "pdfplumber", sdk)
    assert extract_sample_text(path, max_chars=20) == "alphabeta\n"


@pytest.mark.parametrize("text,rtl,script", [("مرحبا", True, "arabic"), ("שלום", True, "hebrew"),
    ("پاکستان", True, "urdu"), ("ܐܒ", True, None), ("English", False, None),
    ("", False, None), ("12345", False, None)])
def test_rtl_language_and_supported_script_detection(text, rtl, script):
    from longparser.utils.rtl_detector import detect_rtl_language, detect_rtl_script

    assert detect_rtl_language(text) is rtl
    assert detect_rtl_script(text) == script


@pytest.mark.parametrize("threshold,expected", [(0.5, True), (0.51, False)])
def test_mixed_rtl_threshold_is_inclusive(threshold, expected):
    from longparser.utils.rtl_detector import detect_rtl_language

    assert detect_rtl_language("hello مرحبا 123", threshold=threshold) is expected


@pytest.mark.parametrize("text,expected", [("", True), ("  " + "x" * 29 + "  ", True),
                                         ("x" * 30, False)])
def test_scan_detection_boundary(text, expected):
    from longparser.utils.ocr_router import is_page_scanned

    assert is_page_scanned(text) is expected


@pytest.mark.parametrize("text,math", [("Ordinary prose", False), ("x = 5", True),
                                     ("α + β", True), ("∫ f(x)", True)])
def test_math_content_detection(text, math):
    from longparser.utils.ocr_router import has_math_content

    assert has_math_content(text) is math


@pytest.mark.parametrize("text,blocks,tables,score", [
    ("Ordinary prose " * 10, 0, False, 0),
    ("Ordinary prose " * 10, 11, False, 1),
    ("Ordinary prose " * 10, 21, False, 2),
    ("x = 5 " * 30, 21, True, 7),
    ("short", 6, False, 1),
    ("Ordinary prose " * 10, 10, True, 3),
])
def test_ocr_complexity_uses_math_tables_and_density(text, blocks, tables, score):
    from longparser.utils.ocr_router import score_page_complexity

    assert score_page_complexity(text, num_blocks=blocks, has_tables=tables) == score


@pytest.mark.parametrize("score,strategy", [(2, "standard"), (3, "math"), (4, "math"), (5, "full_ocr")])
def test_ocr_strategy_thresholds(score, strategy):
    from longparser.utils.ocr_router import get_ocr_strategy

    assert get_ocr_strategy(score) == strategy


def test_default_construction_and_processing_are_license_isolated_in_fresh_process(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    path = tmp_path / "synthetic.txt"
    path.write_text("Body text.")
    script = r'''
import importlib.abc
import socket
import sys
import pytest

blocked = ("pymupdf4llm", "pymupdf", "fitz", "marker", "surya", "torch",
           "longparser.extractors.pymupdf_extractor", "longparser.extractors.marker_extractor")
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            attempts.append(fullname)
            raise AssertionError("Forbidden default-route import: " + fullname)
sys.meta_path.insert(0, Guard())
from tests.pipeline_doubles import install_docling_sdk_stubs
with pytest.MonkeyPatch.context() as patch:
    install_docling_sdk_stubs(patch)
    def deny_network(*args, **kwargs):
        raise AssertionError("Network access forbidden")
    patch.setattr(socket.socket, "connect", deny_network)
    from longparser import DocumentPipeline
    from longparser.extractors.docling_extractor import DoclingExtractor
    from longparser.chunkers import quality_scorer
    from longparser.schemas import ChunkingConfig, Document, DocumentMetadata, Page
    from tests.conftest import make_block
    document = Document(metadata=DocumentMetadata(source_file="synthetic.txt"),
                        pages=[Page(page_number=1, width=100, height=100,
                                    blocks=[make_block("Body text.")])])
    patch.setattr(DoclingExtractor, "extract", lambda self, path, config: (document, None))
    patch.setattr(DoclingExtractor, "get_hierarchy", lambda self, path, config: [])
    patch.setattr(quality_scorer, "_english_words", set())
    pipeline = DocumentPipeline()
    result = pipeline.process_file(sys.argv[1])
    chunks = pipeline.chunk(result, ChunkingConfig(min_tokens=0, overlap_blocks=0))
    assert pipeline._backend_name == "docling"
    assert [chunk.text for chunk in chunks] == ["Body text."]
    assert not attempts, attempts
    loaded = [name for name in sys.modules
              if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)]
    assert not loaded, loaded
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_missing_pymupdf_runs_real_dependency_guard(pipeline_env, monkeypatch):
    monkeypatch.delitem(sys.modules, "longparser.extractors.pymupdf_extractor")
    original_import = builtins.__import__

    def absent(name, *args, **kwargs):
        if name == "pymupdf4llm":
            raise ImportError("Optional SDK deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)
    with pytest.raises(ImportError, match="pymupdf4llm is not installed"):
        pipeline_env.module.PipelineOrchestrator(ProcessingConfig(backend="pymupdf"))


@pytest.fixture
def native_pdf_sdk(monkeypatch, tmp_path):
    """SDK/file boundaries only: execute the real native extractor entry points."""
    sdk = ModuleType("pymupdf4llm")
    sdk.to_markdown = lambda *args, **kwargs: "# First\nFirst page text.\n---\n# Second\nSecond page text."
    monkeypatch.setitem(sys.modules, "pymupdf4llm", sdk)
    state = SimpleNamespace(closed=False)

    class PDF:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return SimpleNamespace(rect=SimpleNamespace(width=600 + index, height=800))

        def close(self):
            state.closed = True

    fitz = ModuleType("pymupdf")
    fitz.open = lambda *args, **kwargs: PDF()
    monkeypatch.setitem(sys.modules, "pymupdf", fitz)
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"Synthetic SDK fixture; not an actual PDF")
    return path, state


def test_real_native_backend_can_construct_and_return_requested_page(native_pdf_sdk):
    from longparser.extractors.pymupdf_extractor import PyMuPDFExtractor

    path, state = native_pdf_sdk
    extractor = PyMuPDFExtractor()
    page = extractor.extract_page(path, 1, ProcessingConfig(export_images=False))
    assert page.page_number == 2
    assert (page.width, page.height) == (601, 800)
    assert [block.text for block in page.blocks] == ["Second", "Second page text."]
    assert all(block.provenance.page_number == 2 and block.provenance.source_file == str(path)
               for block in page.blocks)
    assert state.closed


@pytest.mark.parametrize("index", [-1, 2])
def test_real_native_page_request_out_of_range_raises(native_pdf_sdk, index):
    from longparser.extractors.pymupdf_extractor import PyMuPDFExtractor

    path, _ = native_pdf_sdk
    with pytest.raises(ValueError, match=f"Page {index} not found"):
        PyMuPDFExtractor().extract_page(path, index, ProcessingConfig(export_images=False))
