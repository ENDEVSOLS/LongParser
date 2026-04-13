# Contributing to LongParser

Thank you for your interest! LongParser is an open-source project by ENDEVSOLS and we
welcome contributions of all kinds.

---

## Getting Started

### Prerequisites

- Python 3.10+
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

Use Python 3.10+ type hints. All public API must be fully annotated.

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
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR description explains *what* changed and *why*

---

## Reporting Issues

Use [GitHub Issues](https://github.com/ENDEVSOLS/LongParser/issues).
Include the Python version, LongParser version, and a minimal reproducer.

## Security Vulnerabilities

See [SECURITY.md](SECURITY.md).
