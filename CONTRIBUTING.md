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

## Automatic Releases

After a PR merges into `main`, CI must pass lint, the Python test matrix, and the
package build before releasing. The release job increments the patch version
(for example, `0.1.5` to `0.1.6`) in the package metadata, runtime version, and
lockfile. Do not bump these files manually for routine PRs.

Release notes combine any `[Unreleased]` changelog entries with GitHub's generated
PR notes. PRs without category labels appear under Other Changes. The job writes
a dated changelog section, commits the version changes, creates a tag and GitHub
release, and dispatches the PyPI publishing workflow at that tag. Direct pushes to
`main` do not release. If a newer merge is already on `main`, its successful CI
run releases the combined changes.

The repository must allow GitHub Actions to write to `main` (including any branch
protection or ruleset exceptions), and the existing `pypi` environment and trusted
publisher must permit publication from `publish.yml`. The workflow uses the
built-in `GITHUB_TOKEN`; no personal token is required. A failed API or publishing
step can be retried by rerunning the CI workflow, provided `main` has not advanced.
Publishing runs separately in Actions under **Publish to PyPI**; a successful CI
release job confirms dispatch, while the publishing run confirms the upload.

## Reporting Issues

Use [GitHub Issues](https://github.com/ENDEVSOLS/LongParser/issues).
Include the Python version, LongParser version, and a minimal reproducer.

## Security Vulnerabilities

See [SECURITY.md](SECURITY.md).
