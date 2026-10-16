# Equation context moves without its source page

**Discovered by:** Milestone 3 regression tests

**Status:** Fixed and verified locally for planned release `0.1.6`

**Affected source:** `src/longparser/chunkers/hybrid_chunker.py`, `HybridChunker._pack_blocks`

**Regression:** `tests/unit/test_chunkers.py::test_equation_glue_preserves_cross_page_provenance`

## Reproduction and effect

With `max_tokens=50`, `min_tokens=5`, and `overlap_blocks=0`, supply:

1. `Opening paragraph ` repeated 10 times, page 1.
2. `Equation context words ` repeated 5 times, page 2.
3. `α + β = γ`, typed as an equation, page 3.

The equation exceeds the current packing target. The chunker carries its preceding paragraph into the second chunk. Content and block IDs are correct, but the first chunk reports pages `[1, 2]` and the second reports `[3]`. Correct page lists are `[1]` and `[2, 3]`.

Source citations can therefore point readers to the wrong page or omit the page containing explanatory text. The failure is deterministic and does not involve the embedding model.

## Applied focused repair

The fix records a lookup from source block IDs to source pages during packing. After moving a block, it recomputes the outgoing chunk's pages from its remaining IDs, and adds the carried block's page to the incoming chunk. Recomputing avoids removing a page still shared by another retained block.

The fix preserves text, block IDs, equation grouping, token estimation, overlap, and split decisions. Two regression cases cover separate pages and a shared page. All 46 controlled tests pass against the corrected repository code.

## Release disposition

The user explicitly authorized the focused repair on 7 October 2026 with “fix these errors/issues”. Include this metadata correction in the planned `0.1.6` release as an approved exception to Milestone 3's original unchanged-production scope.

Verification: 120 default-suite tests pass, six real embedding tests pass separately, and configured Ruff passes. The failing regression assertions were retained and neither weakened nor marked as expected failures. Remote CI and maintainer release review remain pending.
