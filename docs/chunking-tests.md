# Chunking and semantic boundary tests

## Two complementary test groups

| Group | What it checks | Embeddings | Default CI |
|---|---|---|---|
| Controlled tests | Exact content, splits, section paths, tables, lists, equation context, provenance, quality scores and semantic edge cases. | Fixed two-dimensional vectors through an external SDK stub; production cosine and packing logic run. | Included; no model download. |
| Real-model tests | Actual model output shape, finite vectors, labelled boundary decisions and resulting chunks. | Pinned `all-MiniLM-L6-v2`, local CPU inference. | Opt-in; explicitly deselected otherwise. |

The controlled group replaces external SentenceTransformer, fastText and selected dictionary resources. Network connections are rejected. All five existing chunker fixtures have meaningful uses, and blocks are fresh or copied before mutation.

The real group replaces only the unrelated fastText scoring resource. SentenceTransformer, embedding vectors, cosine decisions and chunk packing are real. Missing prerequisites fail an explicit run rather than silently skipping it. Runtime network access is rejected in both groups.

## Run the tests

With the existing development/server environment installed:

```bash
.venv/bin/python -m pytest tests/unit/test_chunkers.py --no-cov
.venv/bin/python -m pytest tests/ --cov=longparser --cov-report=term-missing
.venv/bin/ruff check .
```

### Prepare the optional real-model environment

Install CPU PyTorch before the existing embedding extra if needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,server,embeddings-cpu]'
```

Cache the fixed model revision once while network access is available. This preparation is separate from the offline tests:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)
PY
```

Run actual inference:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/integration/test_real_embeddings.py --run-real-embeddings --no-cov
```

The fixture selects the immutable local snapshot with `local_files_only=True`, sets Hugging Face/Transformers offline flags and limits PyTorch CPU threads. The production boundary loader receives that local path. It restores cache/resource/thread state afterwards. Installing the embedding extra alone does not enable these tests.

## Fixed labelled evaluation

[`tests/fixtures/semantic_documents.json`](https://github.com/ENDEVSOLS/LongParser/blob/main/tests/fixtures/semantic_documents.json) contains four three-block documents. Labels and rationales were written before the first model inference:

| Document | Expected boundaries (zero-based, before the block) | Reason |
|---|---|---|
| Identical control | None | Three identical statements. |
| Astronomy → baking | `[2]` | Astronomy paraphrase followed by a cake instruction. |
| Cooking → networks | `[2]` | Cooking paraphrase followed by firewall configuration. |
| Gardening → finance | `[2]` | Gardening paraphrase followed by compound interest. |

Model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41). Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The threshold stays at the production default `0.3`; labels and threshold were not adjusted after inference.

The initial local run matched all eight adjacent-pair labels: three true boundaries, five correct non-boundaries, zero false boundaries and zero missed boundaries. Precision and recall on **this small fixture only** are 100%. Six real tests passed, including vector dimensions `(3, 384)` and chunk packing with source pages/IDs.

This establishes working integration and performance on obvious toy examples. It does not measure retrieval quality, extraction accuracy, general semantic accuracy or performance on representative PDFs. A broader evaluation needs independently labelled documents and retrieval relevance judgements.

## Existing token budget policy (DEC-04)

Preserve `max_tokens` as the existing packing target. Token estimates use whitespace word counts multiplied by `1.33`, with integer rounding; they are not counts from the embedding model tokenizer. Per-block rounding may differ from the joined text count.

| Situation | Current handling | Test/effect |
|---|---|---|
| Divisible normal paragraphs | Start another chunk before the next block exceeds the packing target. | Content, source IDs/pages and expected splits checked. |
| Oversized indivisible paragraph or table row | Retain the complete block or row. | Final chunk can exceed the target; no text silently discarded. |
| Below-minimum chunk | Merge using the existing policy. | Verified content and provenance; merged text can exceed the target. |
| Eligible paragraph overlap | Prepend previous context after packing. | Final count can exceed the target. Source IDs describe primary content; the overlap flag identifies copied context. |
| Table captions/schema chunks | Add context or generate a schema outside normal paragraph packing. | These can exceed the target. Captions on the table's page are tested. |

A strict final maximum would change production behavior and requires separate design and tests. The current milestone documents and preserves existing exceptions; formal release acceptance of DEC-04 remains subject to the recorded scope decision.

## Required-case mapping

| Plan cases | Coverage |
|---|---|
| B01–B04 | Empty, normal, blank/separator and normal budget overflow inputs. |
| B05–B08 | Short-tail merging, supplied hierarchy, positive/zero overlap and section/table exclusions. |
| B09–B12 | Structured row batches, schemas, pipe columns, table-only/fallback text, oversized paragraph/row, heading-only/consecutive headings. |
| B13 | List lead-in and bullet splitting, equation detection/glue and cross-page source references. |
| B14 | Confidence weighting, noise/clamping, dictionary availability/coverage, fastText fallback and unknown sources. |
| S01–S08 | Short input, similar/different vectors, equality threshold, zero norms, missing SDK, disabled semantic mode and blank-filtered semantic packing. |
| Added integration | Pinned real vectors, fixed labels and real-boundary chunk packing. |

Fixed regression and approved release disposition: [equation page provenance issue](issues/equation-page-provenance.md). The authorized page-reference repair passes both regression cases. All 46 controlled tests and all 120 default-suite tests now pass; six real-model tests pass separately.
