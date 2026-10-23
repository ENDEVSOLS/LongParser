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
    """Validate all files, then bump the patch and move unreleased notes into the release."""
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
    if re.search(rf"^## \[{re.escape(version)}\]", content, re.MULTILINE):
        raise ValueError(f"Changelog already contains {version}")
    section = re.search(
        r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if not section:
        raise ValueError("Missing [Unreleased] changelog section")
    pending = section[1].strip().removesuffix("---").rstrip()
    notes = "\n\n".join(part for part in (pending, generated_notes.strip()) if part) + "\n"
    if not notes.strip():
        raise ValueError("Refusing to create a release with empty notes")
    today = datetime.now(UTC).date().isoformat()
    updates[changelog] = (
        content[:section.start()]
        + f"## [Unreleased]\n\n## [{version}] — {today}\n\n{notes}\n---\n\n"
        + content[section.end():]
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
