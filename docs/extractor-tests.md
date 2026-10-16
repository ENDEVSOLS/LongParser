# Milestone 5: Extractor helper and dependency-guard tests

## Outcome

All 62 cases in `tests/unit/test_extractors.py` pass in the regular development/server environment and in a minimal environment without conversion/model packages. The focused file also passes on Python 3.10.21, 3.11.16, 3.12.14 and 3.13.15.

The full default suite passes **287 tests**, with six opt-in real embedding tests deselected. Those six tests pass separately. Configured Ruff and the existing license-workflow tests pass. No production code, public schema types or runtime dependency declarations change in this milestone.

## Required-case mapping

| Plan case | What runs | Evidence |
|---|---|---|
| E01 | Real abstract extractor contract and provenance helper. | Missing `extract_page` prevents construction; a minimal complete subclass provides both methods and expected provenance. |
| E02 | Actual block classification with controlled SDK types and labels. | Headings/titles, paragraphs, tables, lists, figures, captions, headers/footers, equations and code; inferred heading levels and paragraph demotion. |
| E03 | Actual marker recognition/classification. | Numeric, alphabetic, Roman and absent markers. Existing behavior is explicit: `Plain heading` produces marker `Plain`, classified as `other`. |
| E04 | Actual relative font-size grouping. | Empty input, duplicates, separate groups, exact 15% tolerance and just-outside values. |
| E05 | Actual hierarchy mapping and level inference. | Font-level ordering, marker-span parent/child inference, late unnumbered demotion, standard-heading exceptions, supplied SDK paths and SDK failure fallback. |
| E06 | Actual enriched-page garble validation. | Empty and ordinary text return true; `/C0` and `/C1` return false; requested page is forwarded. This checks garbling, not overall page quality. |
| E07 | Real PyMuPDF dependency guards. | Missing `pymupdf4llm` and both `pymupdf`/`fitz` aliases produce installation guidance with the original `ImportError` as the cause. |
| E08 | Real Marker constructor guard. | Missing Marker produces the existing extra-install message and preserves the original cause, without loading Marker. |
| E09 | Real LaTeX OCR setup/availability and singleton behavior. | Lazy per-backend instances, missing/unknown backends, thread configuration, absent local UniMERNet assets and MFD local-path/CPU configuration. |
| E10 | Entire focused module in minimal environments. | Collects and executes without installed Docling, Docling-core, PyMuPDF, PyMuPDF4LLM, Marker, pix2tex, pix2text, torch or UniMERNet distributions. |

## Import and model boundaries

The tests reuse Milestone 4's `tests/pipeline_doubles.py` to supply only the external symbols that the Docling extractor imports. They then execute the real extractor/helper code. Cached module bindings are aligned to the controlled SDK types so collection order cannot select incompatible item classes. Fixture-scoped monkeypatches restore external symbols and singleton state.

Hierarchy input uses small in-memory items with text, bounding boxes, references and controlled external chunk metadata. No converter or hierarchy model is constructed. No fixture documents are needed.

For LaTeX availability, external torch and SDK objects are controlled. The successful pix2tex setup uses a fake external constructor; PIL import is deliberately unavailable before optional pre-warming, and an inference-call assertion proves no inference executes. UniMERNet weight loaders are guarded against execution. MFD path checks create a temporary, locally generated placeholder `.onnx` file containing plain fixture bytes; it is not a model and is never loaded. No third-party fixture content is added.

Network connections are rejected. No weights, real OCR inference or provider calls run. These tests verify helper/setup behavior; they do not certify real SDK/model compatibility, conversion accuracy or OCR performance. A real installation smoke check remains part of release validation.

## Coverage and its limits

Coverage measures all 40 production modules with no new omissions. Milestone 5 raises whole-source coverage from **40.41% to 47.53%**: 2,322 of the same 4,885 statements execute. The enforced minimum rises from the provisional 17% to **45%**, leaving 2.53 percentage points of margin. Full-suite Python matrix validation remains pending in remote CI; local cross-version evidence covers the focused file.

| Extractor module | Covered / measured statements | Coverage |
|---|---:|---:|
| Base extractor | 12 / 14 | 85.71% |
| Docling extractor | 278 / 1,112 | 25.00% |
| LaTeX OCR | 124 / 223 | 55.61% |
| PyMuPDF extractor | 102 / 198 | 51.52% |
| Marker extractor | 20 / 76 | 26.32% |

The **selected Docling helper surface** reaches **199 of 202 statements (98.51%)**. This slice includes classification, marker extraction/classification, font clustering, hierarchy mapping/marker sub-clustering and enriched-page validation. It is measured by intersecting coverage's measured lines with those function ranges; it is not the whole extractor or branch coverage.

Required-case completion and meaningful coverage of this deterministic surface are the milestone target. Full coverage of the large converter is not a target. Untested real conversions, table/image parsing, equation enrichment, model loading and inference remain visible in the denominator rather than being omitted. No additional per-helper percentage gate is introduced.

## DEC-09: Existing return-contract discrepancy

`BaseExtractor.extract` is annotated to return `Document`. Concrete extractors and pipeline callers use `(Document, ExtractionMetadata)`. This mismatch is recorded without changing annotations, schemas or public return values in the testing milestone. A future API/typing repair requires separate review.

Marker packaging, exports, selection policy and real compatibility validation remain deferred under Workstream D. Guard tests do not declare official Marker support.

## Reproduce

From the nested Git repository:

```bash
.venv/bin/python -m pytest tests/unit/test_extractors.py --no-cov
.venv/bin/python -m pytest tests/ --cov=longparser --cov-report=term-missing
.venv/bin/ruff check .
```

The minimal environment uses only Pydantic 2.12.5, pytest 9.0.2, pytest-asyncio 1.3.0 and their lightweight dependencies:

```bash
uv venv /tmp/longparser-extractor-tests --python .venv/bin/python
uv pip install --python /tmp/longparser-extractor-tests/bin/python \
  'pydantic==2.12.5' 'pytest==9.0.2' 'pytest-asyncio==1.3.0'
/tmp/longparser-extractor-tests/bin/python -m pytest tests/unit/test_extractors.py
```

The existing Python 3.13 minimal environment was reused; separate temporary environments verified Python 3.10–3.12. Python-specific Pydantic wheels were downloaded during setup where absent from cache; the test runs themselves stayed offline. Package/runtime remains `0.1.5`, with `0.1.6` the target release. Nothing is pushed or published by these checks.

## Current support minimum

The Python 3.10 results above are historical verification evidence. The upcoming release requires Python 3.11 or newer, with supported CI jobs on Python 3.11–3.13.
