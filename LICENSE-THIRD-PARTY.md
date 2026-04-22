# Third-Party Licenses

LongParser core is licensed under the **MIT License**.

Some **optional** backends and integrations use different licenses.
These packages are **never loaded by default** — they are only imported
when you explicitly install them and select them in your configuration.

## Optional Backend Licenses

| Package | License | Install Command | When Loaded |
|---------|---------|-----------------|-------------|
| `pymupdf4llm` | AGPL-3.0 or Artifex Commercial | `pip install "longparser[pymupdf]"` | Only when you set `backend="pymupdf"` |
| `marker-pdf` | GPL-3.0-or-later | `pip install "longparser[marker]"` | Only when you set `backend="marker"` *(future)* |
| `surya-ocr` | GPL-3.0-or-later | `pip install "longparser[surya]"` | Only when explicitly imported *(future)* |

## Core Dependency Licenses (always installed)

| Package | License | Purpose |
|---------|---------|---------|
| `pydantic` | MIT | Schema validation |
| `docling` | MIT | Default PDF extraction engine |
| `docling-core` | MIT | Docling data models |
| `fast-langdetect` | Apache-2.0 | Document language detection |

## What This Means for You

- **If you only use `pip install longparser`** — everything is MIT or Apache-2.0.
  You can use LongParser in any project (commercial, proprietary, open source).

- **If you install `longparser[pymupdf]`** — the `pymupdf4llm` library is
  AGPL-3.0 licensed. You must comply with AGPL terms for the PyMuPDF component,
  OR purchase a commercial license from [Artifex](https://artifex.com).
  LongParser core code remains MIT.

- **If you install `longparser[marker]`** *(future)* — the `marker-pdf` library
  is GPL-3.0 licensed. You must comply with GPL terms for the Marker component.
  LongParser core code remains MIT.

## License Isolation Guarantee

LongParser uses **lazy imports** to ensure GPL/AGPL packages are never loaded
unless explicitly requested. The following guarantees hold:

1. `import longparser` does NOT import any GPL/AGPL package
2. `from longparser import DocumentPipeline` does NOT import any GPL/AGPL package
3. `DocumentPipeline().process_file("doc.pdf")` does NOT import any GPL/AGPL
   package (uses Docling, which is MIT)
4. GPL/AGPL code is only loaded when you explicitly set `backend="pymupdf"` or
   `backend="marker"` in `ProcessingConfig`
