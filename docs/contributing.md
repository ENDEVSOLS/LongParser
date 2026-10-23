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

Releases respect the review and status-check rules on `main`. After a normal PR
merges and CI passes, the workflow opens a version-update PR from
`automation/release-vX.Y.Z`. It increments the patch version in the package
metadata, runtime version, and lockfile, and updates both changelog copies.
Do not bump these files manually for routine PRs.

Release notes combine any `[Unreleased]` entries with GitHub's generated PR
notes. PRs without category labels appear under Other Changes. Keep the
`[Unreleased]` heading; the release job moves its contents into a dated entry.
If more PRs merge before the version-update PR is approved, the workflow refreshes
that same automation branch and PR. Its checks and approvals must pass again as
required by repository policy.

Review and merge the version-update PR. After CI passes on its merged commit,
the workflow tags that exact commit, creates the GitHub release, and dispatches
PyPI publishing at the tag. It never pushes a commit directly to `main`.
A direct push to `main` does not prepare or publish a release.

### Repository setup

In **Settings → Actions → General → Workflow permissions**, enable **Allow GitHub
Actions to create and approve pull requests**. An organization administrator may
need to allow this setting first. The workflow only creates/updates PRs; human
review and the repository's merge rules still apply. No bypass or personal token
is required.

The workflow explicitly dispatches CI and license checks on its automation branch,
because bot pushes do not start regular push workflows with `GITHUB_TOKEN`.
Keep the existing `pypi` environment and trusted publisher for `publish.yml`.
Publishing runs separately in Actions under **Publish to PyPI**; the CI release
job confirms dispatch, while that publishing run confirms the upload.

Rerun failed CI jobs to retry PR creation or release creation without incrementing
the version twice. A publishing failure can be retried from **Publish to PyPI**.
An existing version tag must point to the exact tested merge commit; the workflow
fails rather than moving a conflicting tag.

## Reporting Issues

Use [GitHub Issues](https://github.com/ENDEVSOLS/LongParser/issues).
Include the Python version, LongParser version, and a minimal reproducer.

## Security Vulnerabilities

See [SECURITY.md](security.md).
