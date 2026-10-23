"""Prepare an automatic patch release without resolving or changing dependencies."""

from __future__ import annotations

import argparse
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def replace_once(pattern: str, replacement: str, content: str) -> str:
    """Require exactly one matching version declaration before replacing it."""
    updated, count = re.subn(pattern, lambda _: replacement, content, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one matching declaration, found {count}: {pattern}")
    return updated


def release_notes(changelog: str, version: str) -> str:
    """Read the exact numbered section for retrying release creation."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing changelog entry for {version}")
    return match[1].strip().removesuffix("---").rstrip() + "\n"


def prepare(root: Path, generated_notes: str, output: Path) -> str:
    """Bump the patch using pending notes or a prewritten numbered changelog entry."""
    project = root / "pyproject.toml"
    package = root / "src/longparser/__init__.py"
    lock = root / "uv.lock"
    changelog = root / "CHANGELOG.md"
    old = tomllib.loads(project.read_text())["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", old):
        raise ValueError(f"Automatic releases require a stable X.Y.Z version, got {old}")
    major, minor, patch = map(int, old.split("."))
    version = f"{major}.{minor}.{patch + 1}"
    escaped = re.escape(old)
    updates = {
        project: replace_once(rf'^version = "{escaped}"$', f'version = "{version}"', project.read_text()),
        package: replace_once(rf'^__version__ = "{escaped}"$', f'__version__ = "{version}"', package.read_text()),
        lock: replace_once(
            rf'(\[\[package\]\]\nname = "longparser"\nversion = "{escaped}")',
            f'[[package]]\nname = "longparser"\nversion = "{version}"',
            lock.read_text(),
        ),
    }
    content = changelog.read_text()
    numbered = list(re.finditer(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        content, re.MULTILINE | re.DOTALL,
    ))
    unreleased = list(re.finditer(
        r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)",
        content, re.MULTILINE | re.DOTALL,
    ))
    if len(numbered) > 1 or len(unreleased) > 1:
        raise ValueError("Duplicate changelog sections for the pending release")
    if numbered and unreleased:
        raise ValueError("Use either a numbered pending release or [Unreleased], not both")
    section = next(iter(numbered or unreleased), None)
    # After a numbered release, later merges can release generated notes without
    # requiring a maintainer to add an Unreleased heading first.
    first_release = re.search(r"^## \[\d+\.\d+\.\d+\]", content, re.MULTILINE)
    if section is not None:
        start, end = section.start(), section.end()
    elif first_release is not None:
        start = end = first_release.start()
    else:
        raise ValueError("Missing [Unreleased] or numbered changelog section")
    pending = section[1].strip().removesuffix("---").rstrip() if section else ""
    notes = "\n\n".join(part for part in (pending, generated_notes.strip()) if part) + "\n"
    if not notes.strip():
        raise ValueError("Refusing to create a release with empty notes")
    today = datetime.now(UTC).date().isoformat()
    if numbered:
        # Keep the maintainer's version heading and date exactly as written.
        updates[changelog] = content[:numbered[0].start(1)] + f"\n{notes}\n---\n\n" + content[end:]
    else:
        updates[changelog] = (
            content[:start]
            + f"## [Unreleased]\n\n## [{version}] — {today}\n\n{notes}\n---\n\n"
            + content[end:]
        )
    docs_changelog = root / "docs/changelog.md"
    if docs_changelog.is_file():
        updates[docs_changelog] = updates[changelog]
    for path, updated in updates.items():
        path.write_text(updated)
    output.write_text(notes)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--notes", type=Path)
    mode.add_argument("--extract", metavar="VERSION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.extract:
        args.output.write_text(release_notes(Path("CHANGELOG.md").read_text(), args.extract))
    else:
        print(prepare(Path.cwd(), args.notes.read_text(), args.output))


if __name__ == "__main__":
    main()
