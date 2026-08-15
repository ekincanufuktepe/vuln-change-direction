from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from .io import read_jsonl, write_json


class GitCommandError(RuntimeError):
    pass


def _run(args: list[str], cwd: Path | None = None, text: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=text, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode("utf-8", errors="replace").strip()
        raise GitCommandError(f"Command failed ({result.returncode}): {' '.join(args)}\n{stderr}")
    return result


def _repo_directory(repositories_dir: Path, project: str, clone_url: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", project).strip("_") or "repository"
    digest = hashlib.sha256(clone_url.encode("utf-8")).hexdigest()[:10]
    return repositories_dir / f"{safe}_{digest}"


def ensure_repository(repository: Path, clone_url: str, clone_missing: bool) -> None:
    if (repository / ".git").is_dir():
        return
    if not clone_missing:
        raise FileNotFoundError(f"Missing local repository {repository}; use --clone-missing to clone it")
    repository.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", clone_url, str(repository)])


def ensure_revision(repository: Path, revision: str) -> None:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=repository, capture_output=True, check=False
    )
    if probe.returncode == 0:
        return
    _run(["git", "fetch", "--no-tags", "origin", revision], cwd=repository)


def read_blob(repository: Path, revision: str, file_path: str) -> str:
    result = _run(["git", "show", f"{revision}:{file_path}"], cwd=repository, text=False)
    return result.stdout.decode("utf-8", errors="replace")


def changed_path_pairs(repository: Path, parent: str, commit: str) -> dict[str, tuple[str, str]]:
    """Return after-path keyed old/new paths for modified or renamed Java files."""
    result = _run(
        ["git", "diff", "--name-status", "-M", parent, commit, "--", "*.java"], cwd=repository
    )
    pairs: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) == 3:
            old_path, new_path = fields[1], fields[2]
            pairs[new_path] = (old_path, new_path)
            pairs.setdefault(old_path, (old_path, new_path))
        elif status.startswith("M") and len(fields) == 2:
            pairs[fields[1]] = (fields[1], fields[1])
        # Added and deleted files are excluded because Coming's current fine-grain
        # analyzer does not compare an empty source file.
    return pairs


def _revision_path(root: Path, sample_id: str) -> Path:
    return root / sample_id


def materialize_samples(
    manifest_path: str | Path,
    repositories_dir: str | Path,
    pairs_dir: str | Path,
    clone_missing: bool = False,
) -> dict:
    """Materialize full Java files in Coming's ``files`` input layout."""
    samples = read_jsonl(manifest_path)
    repositories_root = Path(repositories_dir)
    pairs_root = Path(pairs_dir)
    pairs_root.mkdir(parents=True, exist_ok=True)
    report = {"materialized": [], "failed": []}

    for sample in samples:
        sample_id = sample["sample_id"]
        try:
            repository = _repo_directory(repositories_root, sample["project"], sample["clone_url"])
            ensure_repository(repository, sample["clone_url"], clone_missing)
            ensure_revision(repository, sample["source_revision"])
            ensure_revision(repository, sample["target_revision"])
            path_pairs = changed_path_pairs(
                repository, sample["parent_commit_hash"], sample["commit_hash"]
            )
            revision_root = _revision_path(pairs_root, sample_id)
            files_written = 0
            for index, file_path in enumerate(sample["file_paths"]):
                if file_path not in path_pairs:
                    raise ValueError(f"Selected path is not a modified/renamed Java file: {file_path}")
                old_path, new_path = path_pairs[file_path]
                if sample["source_revision"] == sample["parent_commit_hash"]:
                    source_path, target_path = old_path, new_path
                else:
                    source_path, target_path = new_path, old_path
                source = read_blob(repository, sample["source_revision"], source_path)
                target = read_blob(repository, sample["target_revision"], target_path)
                file_key = f"f{index:04d}"
                file_root = revision_root / sample_id / file_key
                file_root.mkdir(parents=True, exist_ok=True)
                (file_root / f"{sample_id}_{file_key}_s.java").write_text(source, encoding="utf-8")
                (file_root / f"{sample_id}_{file_key}_t.java").write_text(target, encoding="utf-8")
                files_written += 1
            if files_written == 0:
                raise ValueError("No Java file pairs were written")
            report["materialized"].append({"sample_id": sample_id, "files": files_written})
        except Exception as exc:  # keep a complete exclusion audit for the study
            report["failed"].append({"sample_id": sample_id, "error": str(exc)})

    write_json(pairs_root / "materialization_report.json", report)
    return report
