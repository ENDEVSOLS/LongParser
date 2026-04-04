# Extractors Reference

## BaseExtractor

Abstract base class for all document extractors.

```python
from longparser.extractors.base import BaseExtractor
```

All extractors implement:

```python
def extract(self, file_path: Path, config: ProcessingConfig) -> Document:
    ...
```

## DoclingExtractor

Production extractor using Docling with Tesseract CLI OCR.

```python
from longparser.extractors.docling_extractor import DoclingExtractor

extractor = DoclingExtractor(
    tesseract_lang=["eng"],
    tessdata_path=None,         # uses system default
    force_full_page_ocr=False,
)

doc = extractor.extract("report.pdf", ProcessingConfig())
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tesseract_lang` | `list[str]` | `["eng"]` | OCR language codes |
| `tessdata_path` | `str \| None` | `None` | Path to tessdata directory |
| `force_full_page_ocr` | `bool` | `False` | OCR all pages regardless of embedded text |

### Formula Modes

| Mode | Speed | Quality | Description |
|---|---|---|---|
| `fast` | ⚡⚡⚡ | ★★☆ | Unicode normalization only |
| `smart` | ⚡⚡☆ | ★★★ | BBox crop → pix2tex for FORMULA blocks |
| `full` | ⚡☆☆ | ★★★ | Docling enrichment enabled for all pages |

### Supported Formats

| Extension | Notes |
|---|---|
| `.pdf` | Layout analysis, OCR, table structure, optional formula OCR |
| `.docx` | OMML → LaTeX injection via python-docx |
| `.pptx` | Slide-by-slide, python-pptx indent levels |
| `.xlsx` | Sheet-aware, ExcelFormat option |
| `.csv` | Column-type inference, CsvFormat option |

## LaTeXOCR

Optional equation OCR backend.

```python
from longparser.extractors.latex_ocr import LaTeXOCR

ocr = LaTeXOCR(backend="pix2tex")
latex = ocr.recognize(pil_image)  # Returns LaTeX string
```

!!! note
    Requires `pip install "longparser[latex-ocr]"` (`pix2tex`).
