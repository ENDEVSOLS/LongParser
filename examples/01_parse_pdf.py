"""Quick start example — parse a PDF with LongParser.

Run::

    pip install longparser
    python examples/01_parse_pdf.py your_document.pdf

Output:
    - Prints page count and first 3 chunk texts
    - Saves chunks to chunks.json
"""

import json
import sys
from pathlib import Path

from longparser import PipelineOrchestrator, ProcessingConfig


def main(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Processing: {path.name}")

    config = ProcessingConfig(
        do_ocr=True,
        ocr_backend="easyocr",
        formula_mode="smart",
    )

    pipeline = PipelineOrchestrator(config=config)
    result = pipeline.process_file(str(path))

    doc = result.document
    print(f"Pages:  {doc.metadata.total_pages}")
    print(f"Blocks: {len(doc.all_blocks)}")
    print(f"Chunks: {len(result.chunks)}")
    print()

    print("=== First 3 chunks ===")
    for chunk in result.chunks[:3]:
        print(f"[{chunk.chunk_type} | page {chunk.page_numbers}]")
        print(chunk.text[:200])
        print()

    out = Path("chunks.json")
    with out.open("w") as f:
        json.dump(
            [c.model_dump(mode="json") for c in result.chunks],
            f, indent=2
        )
    print(f"Saved {len(result.chunks)} chunks to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_parse_pdf.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
