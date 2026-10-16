# Whole-project test expansion

**Requested target:** 95–100% whole-production statement coverage.

**Starting evidence:** 287 passing default tests, six separate real embedding tests, 47.53% coverage (2,322 / 4,885 statements).

**Status:** Coverage target reached locally: **95.97%** (4,691 / 4,888 statements). All **448 default tests pass**, and all six real embedding tests pass separately. The user approved the remaining repairs, and all four defect regressions now pass.

This extends the earlier five-milestone scope. It includes public SDK conversion/chunking/export interfaces, integration adapters, embedding/vector-store adapters, database and queue interfaces, API endpoints, workers and chat flows. Existing helper tests remain where the original plan explicitly selected those seams.

## Measurement and assertion policy

- Keep all production modules in coverage. Do not add omissions or broad exclusions to achieve the percentage.
- Test successful behavior, invalid inputs, failure handling, data/provenance mapping and tenant boundaries with independently specified expected outputs.
- Control external model/SDK/database/provider interfaces; exercise real LongParser orchestration, transformations and algorithms.
- Run real-model and clean-installed parsing evidence separately. Mocked SDK coverage does not certify external service/model accuracy.
- Preserve current runtime versions and public behavior. Record newly discovered defects explicitly rather than weakening assertions.
- Raise the enforced floor only after the full suite actually reaches it. Remote CI remains separate from local verification.

## Work order

1. Embedding/vector-store operations and queue lifecycle.
2. Database CRUD, review/versioning, session memory and adapters.
3. API upload/review/finalization/retrieval/chat and workers.
4. Chat retrieval, memory budgeting, review graph and callback lifecycle.
5. Conversion/OCR boundaries, format transformations and remaining core edge cases.
6. Full combined checks, honest module-level evidence, build/smoke validation and final review.

## Latest verification

| Check | Result |
|---|---|
| Whole-production statement coverage | **95.97%**, 40 modules; no added omissions or exclusions |
| Enforced default coverage minimum | **95%**, replacing 45%; the measured coverage passes this gate |
| Default suite | **448 passed, 0 failed, 6 deselected** on Python 3.13.15 |
| Real embedding tests | **6 passed** separately with the pinned cached CPU model |
| Ruff | Passed |
| Mypy | 89 remaining findings in 18 files; informational, down from 91 |
| Package artifacts | Wheel and source distribution built; both pass Twine |
| Installed-wheel real DOCX smoke | Passed again with the rebuilt wheel, actual Docling and chunking; no network or optional backend imports |
| Remote CI | Pending; no push or PR performed |

### Resolved regressions and repairs

| Defect | Regressions | Disposition |
|---|---:|---|
| Caller search filters override authenticated tenant/job scope | 1 passing | [User-approved repair applied and tested](issues/search-filter-scope.md) |
| Docling image exporter has no latest conversion result | 2 passing | [User-approved repair applied and tested](issues/docling-conversion-state.md) |
| Smart formula mode references uninitialized OCR when no formula blocks exist | 1 passing | [User-approved repair applied and tested](issues/docling-conversion-state.md) |
| Later wide-table column groups lose values | Both formats now pass | [User-approved repair applied](issues/wide-table-column-banding.md) |

All four previously failing regressions pass against the user-approved production repairs. No failure was skipped or marked expected. The fixes preserve authenticated search scope, initialize and capture image-export state, and initialize formula OCR before either formula source is considered.

## Module-level evidence

| Production module | Covered / measured statements | Coverage |
|---|---:|---:|
| `__init__.py` | 23 / 26 | 88.46% |
| `chunkers/__init__.py` | 2 / 2 | 100.00% |
| `chunkers/hybrid_chunker.py` | 463 / 482 | 96.06% |
| `chunkers/quality_scorer.py` | 58 / 58 | 100.00% |
| `chunkers/semantic_boundary.py` | 34 / 34 | 100.00% |
| `extractors/__init__.py` | 3 / 3 | 100.00% |
| `extractors/base.py` | 12 / 14 | 85.71% |
| `extractors/docling_extractor.py` | 1000 / 1115 | 89.69% |
| `extractors/latex_ocr.py` | 223 / 223 | 100.00% |
| `extractors/marker_extractor.py` | 76 / 76 | 100.00% |
| `extractors/pymupdf_extractor.py` | 193 / 198 | 97.47% |
| `integrations/__init__.py` | 14 / 14 | 100.00% |
| `integrations/langchain.py` | 38 / 38 | 100.00% |
| `integrations/llamaindex.py` | 38 / 38 | 100.00% |
| `pipeline/__init__.py` | 3 / 3 | 100.00% |
| `pipeline/cross_reference.py` | 95 / 99 | 95.96% |
| `pipeline/orchestrator.py` | 176 / 182 | 96.70% |
| `pipeline/pii_redactor.py` | 137 / 144 | 95.14% |
| `pipeline/summary_enricher.py` | 56 / 56 | 100.00% |
| `schemas.py` | 197 / 197 | 100.00% |
| `server/__init__.py` | 2 / 2 | 100.00% |
| `server/app.py` | 411 / 420 | 97.86% |
| `server/chat/__init__.py` | 6 / 6 | 100.00% |
| `server/chat/callbacks.py` | 32 / 32 | 100.00% |
| `server/chat/checkpointer.py` | 22 / 22 | 100.00% |
| `server/chat/engine.py` | 102 / 113 | 90.27% |
| `server/chat/graph.py` | 63 / 63 | 100.00% |
| `server/chat/llm_chain.py` | 35 / 35 | 100.00% |
| `server/chat/retriever.py` | 46 / 48 | 95.83% |
| `server/chat/schemas.py` | 81 / 81 | 100.00% |
| `server/db.py` | 206 / 206 | 100.00% |
| `server/embeddings.py` | 97 / 98 | 98.98% |
| `server/queue.py` | 52 / 52 | 100.00% |
| `server/schemas.py` | 125 / 125 | 100.00% |
| `server/vectorstores.py` | 221 / 221 | 100.00% |
| `server/worker.py` | 226 / 236 | 95.76% |
| `utils/__init__.py` | 4 / 4 | 100.00% |
| `utils/lang_detect.py` | 62 / 64 | 96.88% |
| `utils/ocr_router.py` | 29 / 29 | 100.00% |
| `utils/rtl_detector.py` | 28 / 29 | 96.55% |

## What this verifies

Tests exercise real LongParser conversion mapping, hierarchy, chunk packing, provenance, table bands, formulas, exports, extraction/index workers, embedding/vector adapters, review/versioning, tenant-aware retrieval, chat memory, and the actual LangGraph interrupt/resume flow. External provider/model/database SDK interfaces use controlled outputs; HTTP tests use the real FastAPI application and middleware.

The Motor double verifies application queries, updates and index declarations. It does not establish live MongoDB transaction, TTL or concurrency behavior. Redis/ARQ and vector SDK doubles likewise do not certify deployed services. OCR tests verify setup, validation, geometry, fallbacks and orchestration with controlled inference; real OCR accuracy is not measured. The real embedding evaluation and installed-wheel DOCX smoke test provide separately identified real-model/document evidence.

The remaining 197 uncovered statements chiefly include Docling error paths, safety limits and format variations, with smaller gaps in chat budgeting, worker failures and core edge cases. Whole-project coverage exceeds 95%; individual modules are not all at 95%, and 100% is not claimed.

## Reproduction

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/ --cov=longparser --cov-report=term-missing --cov-report=html
python -m pytest tests/integration/test_real_embeddings.py --run-real-embeddings --no-cov
ruff check .
mypy src/longparser/
```

The default command passes all 448 tests and the 95% coverage gate. Real embedding tests require the documented pinned model in the local cache. The current supported CI matrix is Python 3.11–3.13. Local evidence is retained in `/tmp/longparser-full-project-*.log`, `/tmp/longparser-full-project-coverage.json`, and `/tmp/longparser-full-project-html/index.html`.

## Repair validation

The user explicitly authorized resolving the four failures. The latest full run passes all affected assertions without changing test expectations. The three production additions increase the denominator from 4,885 to 4,888 statements. Coverage remains whole-source statement coverage with no added exclusions. Informational mypy drops from 91 to 89 findings because both uninitialized-local findings are resolved.

## Python support policy

The user requested raising the minimum to Python 3.11 following Python 3.10 end of life on October 1, 2026. Packaging, CI, Ruff, mypy, installation instructions and contributor documentation now use that minimum. The original Python 3.10 CI failure remains undiagnosed because its detailed log was unavailable; removing the job changes the support policy and does not establish a bug fix. Historical test measurements retain the interpreter versions actually used. See the [official Python lifecycle](https://devguide.python.org/versions/).

Validation after the support-policy update:

- Full default suite on Python 3.13.15: **448 passed, 6 opt-in tests deselected**, with **95.97%** whole-source coverage (4,691 / 4,888 statements).
- Focused extractor suite on Python 3.11.16: **62 passed**. This does not replace the full supported CI matrix.
- Ruff, lockfile consistency, wheel/sdist build, Twine metadata checks and strict documentation build passed. The built wheel declares `Requires-Python: >=3.11` and advertises Python 3.11–3.13.
- Informational mypy still reports **89 errors in 18 files**. Remote CI for these new changes remains pending.

The lockfile was regenerated for the new minimum, pruning Python 3.10 resolution branches and selecting compatible dependency versions where necessary. Public schema enums retain their existing formatting behavior; UTC timestamps use Python 3.11's equivalent `datetime.UTC` alias.
