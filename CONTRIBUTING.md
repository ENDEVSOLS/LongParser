# Contributing to LongParser

Thank you for your interest! LongParser is an open-source project by ENDEVSOLS and we
welcome contributions of all kinds.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- MongoDB (local or Docker)
- Redis (local or Docker)

### Setup

```bash
# 1. Fork + clone
git clone https://github.com/ENDEVSOLS/LongParser.git
cd LongParser

# 2. Install all dependencies
uv sync --extra dev

# 3. Copy env config
cp .env.example .env

# 4. Run tests
uv run pytest tests/unit/ -v
```

---

## Development Workflow

### Branch Naming

| Type | Pattern |
|------|---------|
| Feature | `feat/short-description` |
| Bug fix | `fix/issue-number-description` |
| Docs | `docs/what-changed` |
| Refactor | `refactor/what-changed` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Qdrant vector store support
fix: prevent turn overwrite in resume_chat (#42)
docs: add LangChain integration example
test: add unit tests for HybridChunker table splitting
```

---

## Code Standards

### Linting

```bash
uv run ruff check src/ tests/      # lint
uv run ruff format src/ tests/     # auto-format
```

### Docstrings

All **public** classes, functions, and methods must have Google/NumPy-style
docstrings with `Args:` and `Returns:` sections.

### Type Hints

Use Python 3.11+ type hints. All public API must be fully annotated.

---

## Testing

```bash
# Unit tests only (fast, no external services):
uv run pytest tests/unit/ -v

# With coverage:
uv run pytest tests/unit/ --cov=src/longparser --cov-report=term-missing

# Full test suite (requires MongoDB + Redis):
uv run pytest tests/ -v
```

All new features **must** include unit tests. PRs without tests will not be merged.

---

## Pull Request Checklist

- [ ] Tests pass locally (`uv run pytest tests/unit/`)
- [ ] Ruff passes (`uv run ruff check src/ tests/`)
- [ ] New public API has docstrings
- [ ] Optional detailed release notes added under `[Unreleased]` in `CHANGELOG.md`
- [ ] PR description explains *what* changed and *why*

---

## Releases (Owner-Managed)

Releases are managed directly by project maintainers:

1. **Version Update:**
   - Update `version` in `pyproject.toml` and `__version__` in `src/longparser/__init__.py`.
   - Update `uv.lock` by running `uv lock`.
   - Document changes in `CHANGELOG.md` and `docs/changelog.md` under the release version heading.
   - Commit and merge the changes into `main` after CI checks pass.

2. **Publishing to PyPI:**
   - Create and publish a GitHub Release with tag `vX.Y.Z` (e.g., `v0.1.6`) from `main`.
   - The `.github/workflows/publish.yml` workflow automatically triggers when the release is published, running the test suite, building the distribution, and publishing to PyPI using PyPA Trusted Publishing.
   - Alternatively, maintainers can trigger the `Publish to PyPI` workflow manually via GitHub Actions **Run workflow** (`workflow_dispatch`) to test on `testpypi` or publish to `pypi`.

## Reporting Issues

Use [GitHub Issues](https://github.com/ENDEVSOLS/LongParser/issues).
Include the Python version, LongParser version, and a minimal reproducer.

## Security Vulnerabilities

See [SECURITY.md](SECURITY.md).
