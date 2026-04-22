"""Pipeline performance benchmark for regression testing.

Run this BEFORE and AFTER v0.2.x changes to prove no speed regression.

Usage:
    # Save baseline (v0.1.3)
    python tests/benchmarks/benchmark_pipeline.py > benchmark_v013.txt

    # After v0.2.x changes
    python tests/benchmarks/benchmark_pipeline.py > benchmark_v020.txt

    # Compare
    diff benchmark_v013.txt benchmark_v020.txt
"""

import time
import sys
from pathlib import Path


def benchmark_file(file_path: str) -> dict:
    """Benchmark a single file through the pipeline."""
    from longparser import DocumentPipeline, ProcessingConfig

    path = Path(file_path)
    if not path.exists():
        return {"file": file_path, "status": "SKIPPED (file not found)"}

    pipeline = DocumentPipeline()
    config = ProcessingConfig()

    t0 = time.time()
    try:
        result = pipeline.process_file(path, config=config)
        elapsed = time.time() - t0

        return {
            "file": path.name,
            "time_seconds": round(elapsed, 2),
            "total_blocks": result.total_blocks,
            "total_pages": result.document.metadata.total_pages,
            "status": "OK",
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "file": path.name,
            "time_seconds": round(elapsed, 2),
            "status": f"ERROR: {e}",
        }


def main():
    """Run benchmark on all available test fixtures."""
    # Look for test PDFs in common locations
    fixture_dirs = [
        Path("tests/fixtures"),
        Path("tests"),
        Path("uploads"),
    ]

    test_files = []
    for d in fixture_dirs:
        if d.exists():
            test_files.extend(sorted(d.glob("*.pdf")))

    if not test_files:
        print("No PDF test files found in tests/fixtures/ or uploads/")
        print("Place some PDFs there and re-run.")
        sys.exit(1)

    print("=" * 60)
    print("LongParser Pipeline Benchmark")
    print("=" * 60)
    print(f"Files found: {len(test_files)}")
    print()

    results = []
    for f in test_files[:5]:  # Cap at 5 files for reasonable benchmark time
        print(f"Benchmarking: {f.name} ...", end=" ", flush=True)
        result = benchmark_file(str(f))
        results.append(result)
        print(f"{result.get('time_seconds', '?')}s — {result['status']}")

    print()
    print("-" * 60)
    print(f"{'File':<30} {'Time':>8} {'Blocks':>8} {'Pages':>6}")
    print("-" * 60)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['file']:<30} {r['time_seconds']:>7.2f}s {r['total_blocks']:>8} {r['total_pages']:>6}")
        else:
            print(f"{r['file']:<30} {r['status']}")
    print("-" * 60)


if __name__ == "__main__":
    main()
