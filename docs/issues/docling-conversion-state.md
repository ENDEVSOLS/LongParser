# Docling conversion state defects

Status: user-authorized repairs applied and tested; the full 448-test suite passes.

The full-project tests expose two existing failures:

1. `DoclingExtractor.save_images()` raises `AttributeError` before and after extraction because `_last_result` is never initialized or saved. This also breaks pipeline image export.
2. Smart PDF formula mode raises `UnboundLocalError` when the math detector is available and Docling reports no formula items. The OCR variable is initialized only inside the nonempty formula branch.

The applied repair initializes `_last_result` to `None`, records successful extraction output, and initializes formula OCR and its processed counter before either formula source is considered. It preserves the existing converter configuration and formula limits.

Seven affected tests pass in the temporary patched checkout. Regression assertions remain in `tests/unit/test_docling_conversion.py`; they are not skipped or marked expected failures. The user explicitly requested resolving these errors, authorizing the exception. Both image-export regressions and the smart-mode regression pass in the full suite.
