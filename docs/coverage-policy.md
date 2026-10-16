# Coverage measurement and gate policy

## Measurement scope

Measure every Python production module under `src/longparser`. There are no file omissions for server, pipeline, extractors, OCR helpers or integration adapters. `source` already restricts measurement to production code, so an additional `tests/*` omission is unnecessary.

Optional dependencies and model requirements do not justify hiding whole modules: missing runtime paths should remain visible, and helper behavior can be exercised with boundary mocks. Coverage's existing default exclusions remain for abstract declarations and `TYPE_CHECKING` imports; the Milestone 2 report contains 25 such excluded lines.

Coverage is statement coverage, not a measure of document accuracy, model quality or end-to-end service availability. Executing an import does not establish that a module's runtime behavior has been tested.

## Milestone 2 baseline

Measured on Python 3.13.15 with the existing development/server dependency environment, Ruff 0.15.4, mypy 1.20.0 and coverage.py 7.13.5. The same 74 tests pass under both policies, with zero failures or skips and two existing Pydantic deprecation warnings.

| Policy | Files | Measured statements | Covered statements | Coverage | Minimum |
|---|---:|---:|---:|---:|---:|
| Previous omissions | 14 | 1,207 | 305 | 25.27% | 5% |
| Full production scope | 40 | 4,873 | 924 | 18.96% | **17% provisional** |

All 14 previously measured files have exactly the same measured and covered statement counts under the wider policy. The wider report adds 26 files, 3,666 measured statements and 619 covered statements that the old policy hid. No tests were added in Milestone 2: the newly reported execution comes from the existing suite.

## Coverage by module group

Percentages below are evidence, not additional per-module pass/fail gates.

| Module/group | Measured statements | Covered statements | Coverage | File omitted? | Important remaining paths |
|---|---:|---:|---:|---|---|
| Core schemas and package initializers | 225 | 222 | 98.67% | No | Remaining lazy-import and unknown-export paths. |
| Hybrid chunker | 479 | 42 | 8.77% | No | Packing, hierarchy, overlap, structured tables, lists and equation cases (Milestone 3). |
| Quality scorer and semantic boundaries | 92 | 0 | 0.00% | No | Fallbacks, scoring and semantic split decisions (Milestone 3). |
| Extractors and LaTeX helpers | 1,620 | 94 | 5.80% | No | Mocked format conversion, heading/table helpers and optional-backend guards (Milestone 5); real OCR/model paths remain separate. |
| Integration adapters | 90 | 0 | 0.00% | No | Adapter conversion and optional-library boundaries; currently unexecuted. |
| Pipeline, PII, references and summaries | 481 | 35 | 7.28% | No | Backend selection, redaction, references and summaries with boundary mocks (Milestone 4). |
| Server and chat | 1,760 | 504 | 28.64% | No | Routes, persistence, queues, workers and vector-store paths; existing execution includes imports and chat utility tests. |
| Utilities | 126 | 27 | 21.43% | No | Language detection, OCR routing and RTL edge cases. |
| **Total** | **4,873** | **924** | **18.96%** | **No** | Planned tests and remaining service/model paths. |

## Milestone 3 verified measurement

The new controlled tests increase full-source statement coverage to **28.71%** (1,400 of 4,876 statements across the same 40 files). The approved equation page-reference fix adds three measured statements to the Milestone 2 denominator of 4,873; the coverage scope and provisional 17% gate are unchanged.

| Module | Covered / measured statements | Coverage |
|---|---:|---:|
| Hybrid chunker | 428 / 482 | 88.80% |
| Quality scorer | 56 / 58 | 96.55% |
| Semantic boundaries | 34 / 34 | 100.00% |

This measurement comes from **120 passing tests**, including both corrected provenance regression cases, with six opt-in real-model tests deselected. The full pytest suite and coverage floor pass. The separate six-test real-model run also passes; its results are not included in these coverage totals.

See [chunking tests](chunking-tests.md) for commands and the fixed labelled evaluation, and [equation page provenance](issues/equation-page-provenance.md) for the authorized repair and release disposition. Final coverage gate selection remains scheduled after Milestones 3–5. The informational mypy run reports 93 errors in 18 files; the carried-ID guard also resolves the prior optional-ID typing error in the chunker, with no new typing findings from the repair.

## Milestone 4 verified measurement

The new pipeline/privacy/reference/summary/utility tests raise full-source coverage to **40.41%** (1,974 of 4,885 statements across the same 40 production files). The authorized PyMuPDF construction/page-selection and auto-fallback repair adds nine measured statements to the Milestone 3 denominator of 4,876. Coverage exclusions and the 17% provisional floor stay unchanged.

| Module | Covered / measured statements | Coverage |
|---|---:|---:|
| Pipeline orchestrator | 125 / 182 | 68.68% |
| PII redactor | 137 / 144 | 95.14% |
| Cross-reference resolver | 95 / 99 | 95.96% |
| Summary enrichment | 56 / 56 | 100.00% |
| Language helpers | 62 / 64 | 96.88% |
| OCR routing | 29 / 29 | 100.00% |
| RTL helpers | 28 / 29 | 96.55% |

This measurement comes from **225 passing default-suite tests**, including all five repaired PyMuPDF regression cases, with six real-model tests deselected. Both pytest and the coverage floor pass. The separate six-test real-model run also passes and is not included in these coverage totals. The [repair disposition](issues/pymupdf-construction-and-auto-fallback.md) records explicit user authorization. Informational mypy reports 91 remaining errors in 18 files, down from 93; the repair resolves the two prior abstract-construction errors without adding new findings.

[Pipeline test documentation](pipeline-tests.md) records the scope, minimal-environment evidence and reproduction commands. Coverage reflects statement execution, including controlled external resources; it does not prove real PDF/OCR quality, general PII completeness or live provider availability.

## Milestone 5 and combined coverage

Milestone 5 adds 62 focused extractor/helper/guard cases without production changes. The full default suite passes **287 tests**, with six opt-in real-model tests deselected; those six pass separately. The source scope stays at 40 files and 4,885 statements, with **2,322 covered (47.53%)**. No additional coverage omissions or diagnostic overrides are used.

| Module/group | Covered / measured statements | Coverage |
|---|---:|---:|
| Core schemas/package initializers | 222 / 225 | 98.67% |
| Hybrid chunker | 428 / 482 | 88.80% |
| Quality scorer/semantic boundaries | 90 / 92 | 97.83% |
| Extractors/LaTeX helpers | 539 / 1,626 | 33.15% |
| Pipeline/redaction/references/summaries | 416 / 484 | 85.95% |
| Server/chat | 504 / 1,760 | 28.64% |
| Utilities | 123 / 126 | 97.62% |
| Integration adapters | 0 / 90 | 0.00% |
| **Whole source** | **2,322 / 4,885** | **47.53%** |

The selected Docling helper slice has 199 of 202 statements covered (98.51%); the entire Docling extractor is 25.00%. These are different denominators. [Extractor test documentation](extractor-tests.md) records the selected functions, case mapping, return-contract discrepancy and remaining conversion/model paths.

## Historical enforced minimum after Milestone 5

`fail_under = 45` replaces the historical 17% provisional floor. The completed local default-suite measurement is 47.53%; rounding down after allowing roughly two percentage points leaves **2.53 points of headroom** (about 124 statements). This is an evidence-based stronger gate, not a claim of 90% whole-project coverage.

The full suite passes against this committed setting on Python 3.13.15. All 62 focused extractor cases also pass in minimal environments on Python 3.10.21, 3.11.16, 3.12.14 and 3.13.15. That focused matrix does not replace the full development/server CI matrix.

The following checks were recorded under the previous support policy and 45% gate. The current policy and 95% gate are documented below:

1. Validate the 45% floor against the full Python 3.10–3.13 CI matrix and review any environment-dependent coverage changes.
2. Keep the same whole-production denominator and active regression assertions. Do not restore broad omissions or bypass coverage to get a passing result.
3. Re-measure after any further source changes; explain denominator changes separately from added execution. Adjust a gate only with documented evidence and a small justified margin.
4. The [combined review, clean-wheel DOCX smoke check and package/docs builds](combined-validation.md) now pass locally. Repeat the applicable artifact/smoke checks after the separate release version bump and complete the remaining release checks.

Remote CI has not run because the implementation branch has not been pushed. The 45% gate is locally verified; full-matrix and release acceptance are pending. Historical milestone measurements above retain their original denominator and gate values for comparison.

## Informational typing check

CI runs `mypy src/longparser/` after installing `.[dev,server]`. Its named step has `continue-on-error: true`; Ruff and pytest/coverage remain blocking checks. Preserve the existing mypy configuration rather than starting a strict-typing cleanup in this release.

The initial baseline is **94 errors in 19 files**, checked across 41 source files before deleting the empty routers initializer. Deleting that initializer removes one empty file, not typing errors. These are existing findings to make visible, not new failures caused by this milestone.

## Reproducing the measurements

From the repository root, with the development and server dependencies installed:

```bash
.venv/bin/python -m pytest tests/ --cov=longparser --cov-report=term-missing --cov-report=json:coverage.json
.venv/bin/ruff check .
.venv/bin/mypy src/longparser/
```

The mypy command exits nonzero while its baseline findings remain; this is expected for the informational step. Review JSON coverage totals and file sets when comparing measurements, not only the headline percentage.

## Full-project expansion: current 95% minimum

The subsequently requested whole-project test expansion reaches **95.97%** (4,691 / 4,888 statements across the same 40 modules) on Python 3.13.15. The enforced minimum is now **95%**. No new omissions or exclusions were added. The approved wide-table correction keeps its statement count; the subsequent approved Docling repairs add three statements.

This checkpoint has **448 passing tests, zero failures and six real-model tests deselected**. The six real-model tests pass separately. All four previously failing defect regressions pass after the user-approved fixes. The [full-project report](full-project-testing.md) records module-level coverage, newly discovered defects, approval status, SDK boundaries and remaining validation. The earlier milestone measurements and 45% policy above are historical evidence, not the current gate.

## Current Python support policy

The upcoming release requires Python 3.11 or newer. Its supported CI matrix is Python 3.11–3.13; validate the current 95% coverage minimum across those jobs. Earlier Python 3.10 measurements above describe the previous support policy and remain historical evidence.
