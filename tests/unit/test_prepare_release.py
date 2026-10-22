"""Exercise the release CLI on repository fixtures, with no network or git mutation."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare_release.py"


@pytest.fixture
def release_repo(tmp_path):
    (tmp_path / "src/longparser").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "longparser"\nversion = "0.1.5"\n')
    (tmp_path / "src/longparser/__init__.py").write_text('__version__ = "0.1.5"\n')
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "longparser"\nversion = "0.1.5"\n'
        '[[package]]\nname = "other"\nversion = "0.1.5"\n'
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased] — planned 0.1.6\n\n### Fixed\n\n"
        "- A bug.\n\n---\n\n## [0.1.5] — 2026-05-05\n\n- Previous release.\n"
    )
    (tmp_path / "generated.md").write_text("## What's Changed\n* Add feature by @user in #12\n")
    return tmp_path


def run_release(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--output", "notes.md"],
        cwd=root, text=True, capture_output=True, check=False,
    )


def test_patch_release_preserves_history_and_dependencies(release_repo):
    root = release_repo
    result = run_release(root, "--notes", "generated.md")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.6"
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "0.1.6"
    assert (root / "src/longparser/__init__.py").read_text() == '__version__ = "0.1.6"\n'
    packages = tomllib.loads((root / "uv.lock").read_text())["package"]
    assert [(p["name"], p["version"]) for p in packages] == [("longparser", "0.1.6"), ("other", "0.1.5")]
    changelog = (root / "CHANGELOG.md").read_text()
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
            file.write("\n## [0.1.6]\nAlready released.\n")
    else:
        (release_repo / "CHANGELOG.md").write_text("## [Unreleased]\n")
        (release_repo / "generated.md").write_text("")
    before = {p: p.read_bytes() for p in release_repo.rglob("*") if p.is_file()}
    result = run_release(release_repo, "--notes", "generated.md")
    assert result.returncode != 0
    assert {p: p.read_bytes() for p in release_repo.rglob("*") if p.is_file()} == before


@pytest.fixture
def workflow_repo(release_repo, tmp_path_factory):
    """Run the actual release shell against local git and a stub GitHub API."""
    root = release_repo
    remote = tmp_path_factory.mktemp("remote") / "repo.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    for args in (
        ["init", "-b", "main"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.com"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / "scripts").mkdir()
    (root / "scripts/prepare_release.py").write_text(SCRIPT.read_text())
    for args in (["add", "."], ["commit", "-m", "Merge PR"], ["remote", "add", "origin", str(remote)], ["push", "origin", "main"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    bin_dir = tmp_path_factory.mktemp("release-bin")
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "state = pathlib.Path(os.environ['RELEASE_STATE'])\n"
        "if args[0] == 'api':\n"
        "    print(os.environ.get('MERGED', '1') if args[1] != '--method' else '## Changes\\n* Merged PR #12')\n"
        "elif args[:2] == ['release', 'view']:\n"
        "    sys.exit(0 if state.exists() else 1)\n"
        "elif args[:2] == ['release', 'create']:\n"
        "    if os.environ.get('FAIL_CREATE') == '1': sys.exit(1)\n"
        "    state.write_text(args[2])\n"
        "elif args[:2] == ['workflow', 'run']:\n"
        "    assert args[2:] == ['publish.yml', '--ref', 'v0.1.6', '-f', 'target=pypi']\n"
        "    state.with_suffix('.dispatch').write_text(' '.join(args))\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    gh.chmod(0o755)
    workflow = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/auto-tag.yml").read_text())
    shell = next(step["run"] for step in workflow["jobs"]["release"]["steps"] if "run" in step)
    # The production script uses runner-local /tmp paths; isolate each test's notes.
    shell = shell.replace("/tmp/", str(bin_dir) + "/")
    env = dict(os.environ, SOURCE_SHA=sha, GITHUB_REPOSITORY="org/repo",
               GITHUB_OUTPUT=str(bin_dir / "output"), RELEASE_STATE=str(bin_dir / "state"),
               PATH=f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}")
    return root, shell, env


def test_workflow_pushes_release_and_rerun_does_not_bump_again(workflow_repo):
    root, shell, env = workflow_repo
    for _ in range(2):
        result = subprocess.run(["bash", "-c", shell], cwd=root, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert Path(env["RELEASE_STATE"]).read_text() == "v0.1.6"
        assert Path(env["RELEASE_STATE"]).with_suffix('.dispatch').exists()
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "0.1.6"
    remote_tag = subprocess.check_output(["git", "ls-remote", "origin", "refs/tags/v0.1.6"], cwd=root, text=True)
    assert "refs/tags/v0.1.6" in remote_tag
    count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=root, text=True)
    assert count.strip() == "2"


def test_workflow_recovers_after_release_api_failure(workflow_repo):
    root, shell, env = workflow_repo
    failed = subprocess.run(
        ["bash", "-c", shell], cwd=root, env=dict(env, FAIL_CREATE="1"),
        capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert not Path(env["RELEASE_STATE"]).exists()
    retry = subprocess.run(["bash", "-c", shell], cwd=root, env=env, capture_output=True, text=True)
    assert retry.returncode == 0, retry.stderr
    assert Path(env["RELEASE_STATE"]).read_text() == "v0.1.6"
    count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=root, text=True)
    assert count.strip() == "2"


@pytest.mark.parametrize("reason", ["direct_push", "newer_merge"])
def test_workflow_skips_unqualified_or_stale_ci(workflow_repo, reason):
    root, shell, env = workflow_repo
    if reason == "direct_push":
        env["MERGED"] = "0"
    else:
        subprocess.run(["git", "commit", "--allow-empty", "-m", "Newer merge"], cwd=root, check=True, capture_output=True)
    result = subprocess.run(["bash", "-c", shell], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not Path(env["RELEASE_STATE"]).exists()
    assert not Path(env["GITHUB_OUTPUT"]).exists()
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "0.1.5"
