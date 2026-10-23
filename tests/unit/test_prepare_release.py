"""Exercise the release CLI on repository fixtures, with no network or git mutation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare_release.py"


@pytest.fixture(params=["unreleased", "numbered"])
def release_repo(tmp_path, request):
    (tmp_path / "src/longparser").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "longparser"\nversion = "0.1.5"\n')
    (tmp_path / "src/longparser/__init__.py").write_text('__version__ = "0.1.5"\n')
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "longparser"\nversion = "0.1.5"\n'
        '[[package]]\nname = "other"\nversion = "0.1.5"\n'
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        + ("## [0.1.6] — 2026-10-01" if request.param == "numbered" else "## [Unreleased] — planned 0.1.6")
        + "\n\n### Fixed\n\n"
        "- A bug.\n\n---\n\n## [0.1.5] — 2026-05-05\n\n- Previous release.\n"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/changelog.md").write_text("Old website changelog\n")
    (tmp_path / "generated.md").write_text("## What's Changed\n* Add feature by @user in #12\n")
    return tmp_path


def run_release(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--output", "notes.md"],
        cwd=root, text=True, capture_output=True, check=False,
    )


def test_patch_release_preserves_history_and_dependencies(release_repo):
    root = release_repo
    numbered = "## [0.1.6] — 2026-10-01" in (root / "CHANGELOG.md").read_text()
    result = run_release(root, "--notes", "generated.md")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.6"
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "0.1.6"
    assert (root / "src/longparser/__init__.py").read_text() == '__version__ = "0.1.6"\n'
    packages = tomllib.loads((root / "uv.lock").read_text())["package"]
    assert [(p["name"], p["version"]) for p in packages] == [("longparser", "0.1.6"), ("other", "0.1.5")]
    changelog = (root / "CHANGELOG.md").read_text()
    assert (root / "docs/changelog.md").read_text() == changelog
    assert changelog.count("## [0.1.6]") == 1
    if numbered:
        assert "## [0.1.6] — 2026-10-01" in changelog
    else:
        assert "## [Unreleased]\n\n## [0.1.6]" in changelog
    assert "## [0.1.5] — 2026-05-05\n\n- Previous release.\n" in changelog
    notes = (root / "notes.md").read_text()
    assert "- A bug." in notes and "#12" in notes
    assert "Previous release" not in notes
    retry = run_release(root, "--extract", "0.1.6")
    assert retry.returncode == 0, retry.stderr
    assert (root / "notes.md").read_text() == notes


def test_release_without_manual_changelog_entry(release_repo):
    (release_repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    result = run_release(release_repo, "--notes", "generated.md")
    assert result.returncode == 0, result.stderr
    assert "#12" in (release_repo / "notes.md").read_text()


def test_next_release_after_numbered_changelog(release_repo):
    changelog = release_repo / "CHANGELOG.md"
    changelog.write_text(changelog.read_text().replace("## [Unreleased] — planned 0.1.6", "## [0.1.6] — 2026-10-01"))
    assert run_release(release_repo, "--notes", "generated.md").returncode == 0
    previous = changelog.read_text()
    result = run_release(release_repo, "--notes", "generated.md")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.7"
    assert changelog.read_text().endswith(previous[previous.index("## [0.1.6]"):])


def test_repository_release_inputs(tmp_path):
    """Catch post-merge release failures using the real checked-in release inputs."""
    root = SCRIPT.parents[1]
    for name in ("pyproject.toml", "src/longparser/__init__.py", "uv.lock", "CHANGELOG.md", "docs/changelog.md"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    (tmp_path / "generated.md").write_text("## What's Changed\n* Release input validation\n")
    result = run_release(tmp_path, "--notes", "generated.md")
    assert result.returncode == 0, result.stderr
    version = result.stdout.strip()
    assert tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]["version"] == version
    assert (tmp_path / "docs/changelog.md").read_text() == (tmp_path / "CHANGELOG.md").read_text()
    retry = run_release(tmp_path, "--extract", version)
    assert retry.returncode == 0, retry.stderr


@pytest.mark.parametrize("broken", ["package", "lock", "changelog", "duplicate", "empty"])
def test_invalid_release_leaves_all_files_unchanged(release_repo, broken):
    if broken == "package":
        (release_repo / "src/longparser/__init__.py").write_text('__version__ = "0.1.4"\n')
    elif broken == "lock":
        (release_repo / "uv.lock").write_text('version = 1\n')
    elif broken == "changelog":
        (release_repo / "CHANGELOG.md").write_text("# Changelog\n")
    elif broken == "duplicate":
        with (release_repo / "CHANGELOG.md").open("a") as file:
            file.write("\n## [0.1.6]\nDuplicate.\n\n## [0.1.6]\nDuplicate again.\n")
    else:
        (release_repo / "CHANGELOG.md").write_text("## [Unreleased]\n")
        (release_repo / "generated.md").write_text("")
    before = {p: p.read_bytes() for p in release_repo.rglob("*") if p.is_file()}
    result = run_release(release_repo, "--notes", "generated.md")
    assert result.returncode != 0
    assert {p: p.read_bytes() for p in release_repo.rglob("*") if p.is_file()} == before


@pytest.fixture
def workflow_repo(release_repo, tmp_path_factory):
    """Run the real release shell against git with protected-main enforcement."""
    root = release_repo
    remote = tmp_path_factory.mktemp("remote") / "repo.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    for args in (
        ["init", "-b", "main"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.com"],
    ):
        git(root, *args)
    (root / "scripts").mkdir()
    (root / "scripts/prepare_release.py").write_text(SCRIPT.read_text())
    for args in (["add", "."], ["commit", "-m", "Merge PR"], ["remote", "add", "origin", str(remote)], ["push", "origin", "main"]):
        git(root, *args)
    sha = git(root, "rev-parse", "HEAD")
    hook = remote / "hooks/pre-receive"
    hook.write_text(
        '#!/bin/sh\nwhile read old new ref; do\n'
        '  if [ "$ref" = "refs/heads/main" ]; then\n'
        '    echo "GH013: Changes must be made through a pull request" >&2\n'
        '    exit 1\n'
        '  fi\ndone\n'
    )
    hook.chmod(0o755)
    bin_dir = tmp_path_factory.mktemp("release-bin")
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "state = pathlib.Path(os.environ['RELEASE_STATE'])\n"
        "if args[0] == 'api':\n"
        "    if args[1] == '--method': print('## Changes\\n* Merged PR #12')\n"
        "    elif '.head.ref' in args[-1]: print(os.environ.get('RELEASE_PR', '0'))\n"
        "    else: print(os.environ.get('MERGED', '1'))\n"
        "elif args[:2] == ['release', 'view']:\n"
        "    sys.exit(0 if state.exists() else 1)\n"
        "elif args[:2] == ['release', 'create']:\n"
        "    if os.environ.get('FAIL_CREATE') == '1': sys.exit(1)\n"
        "    state.write_text(args[2])\n"
        "elif args[:2] == ['pr', 'list']:\n"
        "    print('23' if state.with_suffix('.pr').exists() else '')\n"
        "elif args[:2] == ['pr', 'create']:\n"
        "    if os.environ.get('FAIL_PR') == '1': sys.exit(1)\n"
        "    assert not state.with_suffix('.pr').exists(), 'duplicate release PR'\n"
        "    state.with_suffix('.pr').write_text(' '.join(args))\n"
        "elif args[:2] == ['pr', 'edit']:\n"
        "    assert args[2] == '23'\n"
        "    state.with_suffix('.edit').write_text(' '.join(args))\n"
        "elif args[:2] == ['workflow', 'run']:\n"
        "    with state.with_suffix('.dispatch').open('a') as f: f.write(' '.join(args) + '\\n')\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    gh.chmod(0o755)
    workflow = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/auto-tag.yml").read_text())
    shell = next(step["run"] for step in workflow["jobs"]["release"]["steps"] if "run" in step)
    shell = shell.replace("/tmp/", str(bin_dir) + "/")
    env = dict(os.environ, SOURCE_SHA=sha, GITHUB_REPOSITORY="org/repo",
               RELEASE_STATE=str(bin_dir / "state"),
               PATH=f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}")
    return root, shell, env


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()


def run_workflow(workflow_repo, **overrides):
    root, shell, env = workflow_repo
    # Match actions/checkout: each job starts from the exact tested source commit.
    git(root, "checkout", "--detach", env["SOURCE_SHA"])
    return subprocess.run(["bash", "-c", shell], cwd=root, env=dict(env, **overrides), capture_output=True, text=True)


def merge_release_pr(workflow_repo):
    """Simulate GitHub accepting an approved release PR, without bypassing the hook."""
    root, _, env = workflow_repo
    git(root, "fetch", "origin", "automation/release-v0.1.6")
    git(root, "checkout", "main")
    git(root, "merge", "--no-ff", "-m", "Merge version-update PR", "FETCH_HEAD")
    env["SOURCE_SHA"] = git(root, "rev-parse", "HEAD")
    env["RELEASE_PR"] = "1"
    # GitHub's server-side merge updates main after its rules pass.
    remote = git(root, "remote", "get-url", "origin")
    git(root, f"--git-dir={remote}", "fetch", str(root), f"{env['SOURCE_SHA']}:refs/heads/main")


def test_workflow_opens_release_pr_without_pushing_protected_main(workflow_repo):
    root, _, env = workflow_repo
    result = run_workflow(workflow_repo)
    assert result.returncode == 0, result.stderr
    state = Path(env["RELEASE_STATE"])
    assert state.with_suffix('.pr').exists()
    assert not state.exists(), "Publishing must wait for review and merge"
    assert "refs/tags/v0.1.6" not in git(root, "ls-remote", "origin", "refs/tags/v0.1.6")
    assert env["SOURCE_SHA"] in git(root, "ls-remote", "origin", "refs/heads/main")
    assert "automation/release-v0.1.6" in git(root, "ls-remote", "origin", "refs/heads/automation/release-v0.1.6")
    checks = state.with_suffix('.dispatch').read_text()
    assert "workflow run ci.yml --ref automation/release-v0.1.6" in checks
    assert "workflow run license-check.yml --ref automation/release-v0.1.6" in checks
    assert "publish.yml" not in checks
    retry = run_workflow(workflow_repo)
    assert retry.returncode == 0, retry.stderr
    assert state.with_suffix('.edit').exists()


def test_workflow_publishes_merged_release_pr_and_retries_same_tag(workflow_repo):
    result = run_workflow(workflow_repo)
    assert result.returncode == 0, result.stderr
    merge_release_pr(workflow_repo)
    root, _, env = workflow_repo
    for _ in range(2):
        result = run_workflow(workflow_repo)
        assert result.returncode == 0, result.stderr
        assert Path(env["RELEASE_STATE"]).read_text() == "v0.1.6"
    assert git(root, "rev-parse", "refs/tags/v0.1.6") == env["SOURCE_SHA"]
    assert env["SOURCE_SHA"] in git(root, "ls-remote", "origin", "refs/tags/v0.1.6")
    assert "workflow run publish.yml --ref v0.1.6 -f target=pypi" in Path(env["RELEASE_STATE"]).with_suffix('.dispatch').read_text()
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "0.1.6"


def test_workflow_recovers_after_release_api_failure(workflow_repo):
    assert run_workflow(workflow_repo).returncode == 0
    merge_release_pr(workflow_repo)
    failed = run_workflow(workflow_repo, FAIL_CREATE="1")
    assert failed.returncode != 0
    root, _, env = workflow_repo
    assert not Path(env["RELEASE_STATE"]).exists()
    retry = run_workflow(workflow_repo)
    assert retry.returncode == 0, retry.stderr
    assert Path(env["RELEASE_STATE"]).read_text() == "v0.1.6"
    assert git(root, "rev-parse", "refs/tags/v0.1.6") == env["SOURCE_SHA"]


def test_workflow_recovers_after_pr_creation_failure(workflow_repo):
    failed = run_workflow(workflow_repo, FAIL_PR="1")
    assert failed.returncode != 0
    retry = run_workflow(workflow_repo)
    assert retry.returncode == 0, retry.stderr
    _, _, env = workflow_repo
    assert Path(env["RELEASE_STATE"]).with_suffix('.pr').exists()
    assert not Path(env["RELEASE_STATE"]).exists()


def test_workflow_refreshes_release_pr_after_another_merge(workflow_repo):
    assert run_workflow(workflow_repo).returncode == 0
    root, _, env = workflow_repo
    git(root, "checkout", "main")
    git(root, "commit", "--allow-empty", "-m", "Another approved merge")
    env["SOURCE_SHA"] = git(root, "rev-parse", "HEAD")
    remote = git(root, "remote", "get-url", "origin")
    git(root, f"--git-dir={remote}", "fetch", str(root), f"{env['SOURCE_SHA']}:refs/heads/main")
    result = run_workflow(workflow_repo)
    assert result.returncode == 0, result.stderr
    assert Path(env["RELEASE_STATE"]).with_suffix('.edit').exists()
    git(root, "fetch", "origin", "automation/release-v0.1.6")
    assert git(root, "rev-parse", "FETCH_HEAD^") == env["SOURCE_SHA"]


def test_workflow_rejects_conflicting_existing_tag(workflow_repo):
    assert run_workflow(workflow_repo).returncode == 0
    root, _, env = workflow_repo
    git(root, "tag", "v0.1.6", env["SOURCE_SHA"])
    merge_release_pr(workflow_repo)
    result = run_workflow(workflow_repo)
    assert result.returncode != 0
    assert "already points to a different commit" in result.stdout
    assert not Path(env["RELEASE_STATE"]).exists()


@pytest.mark.parametrize("reason", ["direct_push", "newer_merge"])
def test_workflow_skips_unqualified_or_stale_ci(workflow_repo, reason):
    root, _, env = workflow_repo
    if reason == "direct_push":
        env["MERGED"] = "0"
    else:
        git(root, "commit", "--allow-empty", "-m", "Newer merge")
        remote = git(root, "remote", "get-url", "origin")
        git(root, f"--git-dir={remote}", "fetch", str(root), f"{git(root, 'rev-parse', 'HEAD')}:refs/heads/main")
    result = run_workflow(workflow_repo)
    assert result.returncode == 0, result.stderr
    assert not Path(env["RELEASE_STATE"]).exists()
    assert not Path(env["RELEASE_STATE"]).with_suffix('.pr').exists()
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "0.1.5"
