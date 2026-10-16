# Milestone 4: Pipeline and privacy tests

## Scope and status

`tests/unit/test_pipeline.py` adds 105 controlled cases for the existing pipeline, privacy features, references, summaries and utilities. The user-authorized PyMuPDF repair is applied in the repository. All **105 focused cases pass**, including the five original regression failures; Milestone 4 is complete and verified locally.

The complete default suite has **225 passed and six real-embedding tests deselected**, with zero failures. The six opt-in real embedding tests pass separately. Configured Ruff passes. A minimal-dependency environment also passes all 105 focused cases against the repaired repository.

## Before and after

| Area | Before this milestone | Added verification | Effect |
|---|---|---|---|
| Extractor routing | Backend selection and optional fallbacks lacked focused tests. | Default/explicit/auto/unknown selection, PDF text-length boundaries and real dependency guards. | Exposes the actual PyMuPDF construction failure that fake routing alone hides. |
| Extraction → chunking | No focused pipeline wiring proof. | Simulated extraction feeds real chunking with content, hierarchy, source IDs and pages checked. | Detects lost information between stages without conversion models. |
| PII redaction | No focused pattern/report consistency suite. | Synthetic email, phone, SSN, card and IP cases; NER fallback and supported entity replacement. | Visible placeholders and retained originals are checked together. |
| References | No focused target-selection checks. | Explicit labels, nearest before/after targets, unresolved links and a label-index operation guard. | Documents actual document-order proximity and prevents invented links. |
| Summaries | No focused asynchronous behavior tests. | Fake LLM/DB/tokenizer exercise section grouping, skipped types, truncation, failure and concurrency. | Checks orchestration without provider requests or database sessions. |
| Utilities | Sparse language/RTL/OCR coverage. | Confidence thresholds, fallback readers, supported scripts, scan boundaries and OCR strategies. | Pins current decisions and fallbacks. |

## Required-case mapping

| Plan cases | Evidence |
|---|---|
| P01–P03 | Default Docling, explicit backends, safe auto initialization and recorded backend state. |
| P04 | Native/scanned/non-PDF routing, threshold at 100/101 characters and actual missing-SDK fallback regression. |
| P05 | Unknown strings retain Docling fallback under DEC-03; no new rejection policy. |
| P06 | Fresh-process construction, processing and chunking reject forbidden imports and network connections. |
| P07 | Root and pipeline `DocumentPipeline` aliases are the actual `PipelineOrchestrator`. |
| P08 | Mocked extraction/hierarchy reaches real chunking with source content and metadata retained. |
| R01–R06 | Synthetic/reserved pattern values, Luhn-valid/invalid dummy cards and valid/invalid IPv4 octets. |
| R07–R08 | Blank/clean input, redaction off by default and redaction before chunking when enabled. |
| R09 | Missing spaCy and unavailable local model retain regex-only behavior without downloading. |
| R10 | Mixed patterns have exact placeholders, counters and original-value mappings. |
| References | Explicit “Figure 3”, implicit table/figure direction, nearest candidates, unknown anchor/target and empty inputs. |
| Summaries | Grouped section/equation text, skipped tables/figures/existing summaries, root path, provider/model metadata, failure isolation and request concurrency. |
| Language | High/low/equal-threshold confidence, short input, missing/failing detector, language mapping, reader fallbacks and pipeline language precedence. |
| RTL | Arabic, Hebrew, Urdu, generic Syriac RTL, mixed text, inclusive threshold and empty/numeric input. |
| OCR | Scan threshold, mathematical content, tables, dense blocks and strategy boundaries 2/3 and 4/5. |
| Repair regressions | Real native backend construction, requested-page dimensions/provenance and two out-of-range requests. |

## Test boundaries and privacy limits

Network connections are rejected. External Docling SDK imports use `tests/pipeline_doubles.py`; extractor responses are controlled. Production pipeline routing, redaction, scoring, chunking, reference resolution, summary orchestration and utility functions run normally. No real OCR models execute.

Summary tests replace the provider factory, external messages/tokenizer and asynchronous database response. They do not certify provider SDK compatibility or real database persistence. Empty/irrelevant input still constructs the current factory, but makes no LLM request. Assertions compare sections independently of asynchronous completion order.

PII values are synthetic or reserved. Original values remain in `Block.pii_redactions` for authorized review. Tests confirm that the exercised redaction reports and fallback log paths do not contain those originals. Visible redaction does not delete or encrypt originals or guarantee that every possible sensitive format will be detected.

DEC-08 preserves document block order for implicit proximity. The operation-count test checks that explicit target labels are read once per target even with 32 referring chunks. It does not establish universal `O(N)` behavior: the existing implicit path rebuilds a position index per lookup.

Generic RTL detection recognizes more ranges than specific script classification. For example, the tested Syriac characters produce RTL `True` and script `None`; the tests preserve that distinction.

## Reproduce the checks

From the nested Git repository:

```bash
.venv/bin/python -m pytest tests/unit/test_pipeline.py --no-cov
.venv/bin/python -m pytest tests/ --cov=longparser --cov-report=term-missing
.venv/bin/ruff check .
```

These pytest commands pass against the corrected repository. The original failing assertions remain active; no expected-failure markers or exclusions were added.

The focused module was also run in a separate temporary environment containing only Pydantic 2.12.5, pytest 9.0.2, pytest-asyncio 1.3.0 and their lightweight dependencies:

```bash
uv venv /tmp/longparser-milestone4-lean --python .venv/bin/python
uv pip install --python /tmp/longparser-milestone4-lean/bin/python \
  'pydantic==2.12.5' 'pytest==9.0.2' 'pytest-asyncio==1.3.0'
/tmp/longparser-milestone4-lean/bin/python -m pytest tests/unit/test_pipeline.py
```

The local setup installed from the existing cache, without network access. There are no Docling, PyMuPDF, torch, spaCy, tokenizer, LLM/provider or database packages in this environment. It passes all 105 focused cases against the corrected repository, confirming dependency isolation.

## Approved repair and limits

See [PyMuPDF construction and auto fallback](issues/pymupdf-construction-and-auto-fallback.md) for the defects, applied implementation, user authorization and passing verification. The single-page method extracts the whole file before selecting a page, preserving original dimensions and source references; subset extraction efficiency remains separate work. Package/runtime remains `0.1.5`; target release remains `0.1.6`. No push, PR or merge has occurred.
