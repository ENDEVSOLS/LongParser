# PyMuPDF construction and automatic fallback

**Discovered by:** Milestone 4's actual backend-constructor test.

**Status:** Fixed in the repository and verified locally.

**Target release:** `0.1.6`; focused repair explicitly authorized by the user.

## Original defects and impact

1. Before the repair, `PyMuPDFExtractor` inherited the abstract `extract_page` method without implementing it. Python rejected construction with `TypeError` before the optional SDK guard executed. Explicit `backend="pymupdf"` and native-PDF auto routing failed even with the SDK present.
2. After satisfying that abstract contract in a temporary copy, testing exposed a second defect: `auto` mode propagated the constructor's missing-SDK `ImportError`. Importing the wrapper module succeeded without the SDK, so the intended Docling fallback did not execute. The applied repair now handles this path.

Fake extractor routing tests alone cannot prove either path correct. Five retained regression cases cover automatic missing-SDK fallback, explicit missing-SDK diagnostics, successful real-backend construction/page selection, and two out-of-range page requests. All five initially failed at the abstract-class error. They now pass against the corrected repository.

## Applied focused repair

The repair is applied in `src/longparser/extractors/pymupdf_extractor.py` and `src/longparser/pipeline/orchestrator.py`.

- Implement `extract_page` by extracting the document and selecting the page whose one-based `Page.page_number` matches the requested zero-based index. Raise `ValueError` for a missing page. Using full extraction preserves original page numbers, dimensions and block provenance; it does not assume that the existing subset conversion correctly renumbers pages.
- Begin automatic processing with the existing Docling extractor. Catch only constructor `ImportError` when attempting PyMuPDF; retain Docling and run its normal language-resolution path. Other errors remain visible. Explicit PyMuPDF selection continues to raise its dependency error.

The single-page implementation parses the whole file before selection. This prioritizes correctness over single-page extraction efficiency; optimizing the subset conversion is separate work. It uses controlled SDK responses in tests, not real PDF/OCR conversion.

Repository verification passes all 225 default-suite tests and all 105 focused Milestone 4 cases. The focused cases also pass in a minimal environment without Docling, PyMuPDF, torch, spaCy, tokenizer assets, LLM/provider or database packages. Six real embedding tests pass separately; configured Ruff passes. Remote CI has not run.

## Release disposition

The implementation plan states: **“Product files: Behavior unchanged, except separately approved repairs in their own PRs.”** The user-selected submission workflow uses separate local commits and one final implementation PR; a repair should therefore be isolated in its own local commit if approved.

The user explicitly authorized the repair on 7 October 2026: “resolve all these failure and then test it”. Include the compatible repair in planned release `0.1.6`, isolated in a focused local commit. All five regression assertions remain active and pass. No external issue or message has been sent.

## Applied production changes

```diff
--- src/longparser/extractors/pymupdf_extractor.py
+++ src/longparser/extractors/pymupdf_extractor.py
@@ -205,6 +205,19 @@
         )

         return document, meta
+
+    def extract_page(
+        self,
+        file_path: Path,
+        page_number: int,
+        config: ProcessingConfig,
+    ) -> Page:
+        """Return a 0-indexed page with its original dimensions and provenance."""
+        document, _ = self.extract(file_path, config)
+        for page in document.pages:
+            if page.page_number == page_number + 1:
+                return page
+        raise ValueError(f"Page {page_number} not found in {file_path}")

     def _markdown_to_document(
         self,
--- src/longparser/pipeline/orchestrator.py
+++ src/longparser/pipeline/orchestrator.py
@@ -205,17 +205,20 @@
         logger.info(f"Processing: {file_path.name}")

         # Auto-mode: decide backend per document
+        extractor = self.extractor
         if self._backend_name == "auto" and self._should_use_pymupdf(file_path):
             from ..extractors.pymupdf_extractor import PyMuPDFExtractor
-            extractor = PyMuPDFExtractor()
-            logger.info("Auto mode selected: PyMuPDF4LLM (native PDF detected)")
-        else:
-            extractor = self.extractor
-
-            # Resolve languages for Docling backend
-            if isinstance(extractor, DoclingExtractor):
-                resolved_langs = self._resolve_languages(file_path, config)
-                extractor._languages = resolved_langs
+            try:
+                extractor = PyMuPDFExtractor()
+            except ImportError:
+                logger.debug("Auto mode: PyMuPDF dependency unavailable, using Docling")
+            else:
+                logger.info("Auto mode selected: PyMuPDF4LLM (native PDF detected)")
+
+        # Resolve languages for Docling, including the missing-extra fallback.
+        if isinstance(extractor, DoclingExtractor):
+            resolved_langs = self._resolve_languages(file_path, config)
+            extractor._languages = resolved_langs

         # Extract document
         document, meta = extractor.extract(file_path, config)
```
