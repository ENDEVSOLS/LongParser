# Combined implementation review and local validation

**Verified:** 7 October 2026

**Baseline:** `58c688f` (`0.1.5`)

**Reviewed milestone head:** `ce4de26`, plus the README and validation follow-up changes

**Current package/runtime:** `0.1.5`; target release: `0.1.6`

**Historical Milestone 5 baseline:** The requested combined review, package/documentation builds, clean installation and selected real parsing smoke check are complete locally. Remote CI, submission and release work remain separate.

## Results

| Check | Result | Scope |
|---|---|---|
| Standards review | Passed after one documentation repair | All eight milestone/repair commits and the README changes. |
| Specification review | No blocking mismatch found | Five milestones and approved repairs against the adjusted implementation plan. |
| Full default suite | 287 passed; 6 deselected | Python 3.13.15; two existing Pydantic deprecation warnings. |
| Whole-production coverage | 47.53%; 45% minimum passed | 2,322 of 4,885 statements across 40 modules. |
| Real embedding tests | 6 passed separately | Pinned local MiniLM model; one existing Pydantic warning. |
| Minimal-environment pipeline/extractor tests | 167 passed | Real helper algorithms with controlled SDK boundaries; no conversion/model packages. |
| Ruff | Passed | Full configured repository check. |
| License checks | Passed | Actual workflow static scan and regression cases in the full suite. |
| Informational mypy | 91 existing errors in 18 files | Non-blocking; 40 source files checked. |
| Package build | Passed | Wheel and source distribution; wheel built from the source distribution. |
| Twine metadata checks | Both artifacts passed | Local metadata/README validation; no upload. |
| Documentation | Strict build passed | Existing Material/MkDocs configuration and docs workflow tooling. |
| Fresh core dependency installation | Passed | Built wheel, no server extras, 129 compatible installed distributions. |
| Real extraction and chunking | Passed | Generated original DOCX; actual Docling converter and LongParser chunker. |

The earlier focused extractor matrix passed on Python 3.10–3.13. This combined full-suite run is on Python 3.13; it does not replace the pending full remote CI matrix.

## Standards

The independent standards review checked the combined diff against `CONTRIBUTING.md` and the configured project standards. It found one low-severity issue: the new public `PyMuPDFExtractor.extract_page()` method lacked the required `Args` and `Returns` documentation. Its docstring now explains input indexing, original returned provenance, invalid-page errors and whole-document conversion. Runtime behavior is unchanged.

No actionable new design smell was identified. Mechanical cleanup remains separate from test additions and approved behavior repairs in the existing commit history. The acknowledged whole-document conversion used by `extract_page()` remains an efficiency limitation outside this maintenance scope.

## Specification

The independent specification review compared implementation with the adjusted `0.1.6` plan and original handover. The B/S/P/R/E requirements map to meaningful assertions against real production logic. The equation provenance and PyMuPDF construction/fallback changes have explicit user authorization and active regressions. The real embedding tests were separately requested. No blocking implementation mismatch or unapproved feature expansion was found.

The existing token packing exceptions, unknown-backend fallback, block-order reference proximity and extractor annotation discrepancy remain documented. The review does not turn them into new public contracts or silently repair them. Marker support remains deferred beyond `0.1.6`.

## Documentation corrections

- The README and docs quickstart now call `pipeline.chunk()` after extraction and guard empty results.
- The docs quickstart uses `result.chunks` rather than an undefined `doc` variable and starts the background worker with the API services.
- The site changelog now matches the repository changelog, including the planned release and final coverage floor.
- A fixture link outside the documentation tree now points to its repository location, fixing the strict build warning.
- Test/coverage/repair records are included in the documentation navigation.
- The README reports completed local builds and smoke evidence while preserving pending remote/release status.

## Artifact and installation checks

The build produces `longparser-0.1.5-py3-none-any.whl` and `longparser-0.1.5.tar.gz`. Retaining `0.1.5` is intentional: the version bump belongs to the separate release PR after maintainer merge.

Artifact inspection verifies the declared version, updated README metadata, all production Python modules, `py.typed`, and removal of the empty routers package. Twine validates both artifacts. No package is uploaded.

A newly created virtual environment installs the built wheel and its complete declared core dependency graph, constrained to the already tested local versions. No `--no-deps` substitute is used for dependency resolution. Installation needs a few previously uncached wheels; setup downloads are separate from the offline smoke run. To fit the temporary filesystem quota, dependency files use uv's supported cache symlinks and the LongParser wheel is copied into the environment. The environment has no system-site packages or editable source install.

`uv pip check` reports all 129 distributions compatible. FastAPI, Uvicorn, ARQ, Marker, PyMuPDF4LLM, spaCy and pix2tex are absent. Basic `import longparser` does not initialize Docling, torch, server modules or optional GPL/AGPL backends. The package resolves from the fresh environment's `site-packages`, not the checkout.

## Real parsing smoke check

The [manual smoke script](https://github.com/ENDEVSOLS/LongParser/blob/main/scripts/verify_core_smoke.py) generates a small original DOCX with a title, two section headings and two policy paragraphs. It uses actual installed Docling 2.75.0 and the real `DocumentPipeline`; no extractor, converter or chunking function is mocked.

Processing options disable OCR, equation enrichment, image export and automatic language detection for this text-only fixture. Semantic chunking stays disabled. Network connections are blocked and Hugging Face offline flags are set; no connection attempt occurs during the completed run.

| Observed output | Result |
|---|---|
| Logical document page groups | 1, with existing Docling page index `0` |
| Extracted blocks | 5 |
| Hierarchy items | 2 |
| Explicitly generated chunks | 1 |
| Expected paragraphs | Both preserved in blocks and chunk text |
| Provenance | Source path retained; block references match their document pages; chunk IDs/pages match their source blocks |
| Optional GPL/AGPL backend imports | None |
| Editable installation | None |

This Docling path converts page references to zero-based indices; the smoke assertion checks consistency with actual returned document pages. No production indexing behavior changed. DOCX logical grouping is not proof of rendered Word pagination.

The run emits existing small-chunk merge and fast-langdetect advisory messages; they do not prevent preservation of expected content or valid source references. This verifies one real text-only DOCX path. It does not certify PDF layout/OCR models, every format, live providers, retrieval quality or deployed services. Those broader validations are outside the requested selected-path smoke check.

## Reproduce

Use isolated build/docs tooling installed as in the existing docs workflow (`mkdocs-material`, `mkdocstrings[python]`, `mike`), plus `build`, `setuptools`, `wheel` and `twine`. Runtime dependencies remain unchanged.

The completed final build uses build 1.6.1, setuptools 84.0.0 and wheel 0.48.0. Twine is 7.0.0, MkDocs 1.6.1 and Material 9.7.7. The build runs in fresh isolated build environments with uv as the installer, using cached build dependencies:

```bash
UV_OFFLINE=1 UV_CACHE_DIR=/tmp/longparser-final-validation/uv-cache \
  /tmp/longparser-final-validation/tools-env/bin/python -m build \
  --installer uv --outdir /tmp/longparser-final-validation/artifacts
```

The equivalent ordinary commands below use whichever isolated tooling environment you prepared.

```bash
python -m build --outdir /tmp/longparser-artifacts
twine check /tmp/longparser-artifacts/longparser-0.1.5-py3-none-any.whl \
  /tmp/longparser-artifacts/longparser-0.1.5.tar.gz
mkdocs build --strict --site-dir /tmp/longparser-site
python -m pytest tests/ --cov=longparser --cov-report=term-missing
ruff check .
```

For the wheel smoke check, create a fresh environment, install CPU PyTorch and the wheel with its dependencies, then run the repository script with that environment's Python:

```bash
python -m venv /tmp/longparser-core-smoke
/tmp/longparser-core-smoke/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
/tmp/longparser-core-smoke/bin/python -m pip install \
  /tmp/longparser-artifacts/longparser-0.1.5-py3-none-any.whl
/tmp/longparser-core-smoke/bin/python scripts/verify_core_smoke.py \
  --expected-version 0.1.5 --output-dir /tmp/longparser-smoke-output
```

The completed local run uses exact constraints from the tested dependency environment, including Docling 2.75.0, Docling-core 2.66.0 and CPU torch 2.10.0. Future unconstrained installs may select newer dependencies; a passing run here does not certify every future resolution.

Local logs, coverage JSON, dependency constraints, artifact hashes and smoke JSON are retained under `/tmp/longparser-final-validation`. This directory is a temporary verification artifact, not a published release.

## Remaining work

1. Commit the final documentation/validation follow-up as appropriate; preserve the earlier user `.gitignore` change separately.
2. Push the implementation branch and open its single combined PR; verify the current supported remote Python 3.11–3.13 CI matrix and submit final evidence.
3. Obtain maintainer review and merge.
4. Prepare the separate release PR: matched `0.1.6` versions, final release changelog, committed roadmap/scope record, release artifact/import/parse checks and publication prerequisites.
5. Verify the published artifact after release. Marker support remains deferred.

**Review totals:** Standards: one documentation finding, resolved. Specification: zero blocking implementation findings. The requested local validation is complete; no push, PR, merge or publication occurred.

## Subsequent full-project expansion

The figures above describe the completed milestone baseline. See the [full-project testing report](full-project-testing.md) for the subsequent 95.97% coverage measurement, 95% gate and user-approved defect repairs. It supersedes the baseline as the current test status.
