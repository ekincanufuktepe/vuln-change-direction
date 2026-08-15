from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .constants import FIXING, INTRODUCING, OBSERVED_FIX, SYNTHETIC_INVERSE
from .io import read_json, write_jsonl

'''
def clone_url_from_commit_url(commit_url: str, repo_name: str) -> str:
    """Derive a public clone URL from a MegaVul commit URL."""
    if commit_url:
        parsed = urlsplit(commit_url)
        path = parsed.path.rstrip("/")
        for marker in ("/-/commit/", "/commit/", "/commits/"):
            if marker in path:
                path = path.split(marker, 1)[0]
                break
        if parsed.scheme and parsed.netloc and path:
            if not path.endswith(".git"):
                path += ".git"
            return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    if repo_name.count("/") == 1:
        return f"https://github.com/{repo_name}.git"
    raise ValueError(f"Cannot derive clone URL for repository {repo_name!r}")
'''
from urllib.parse import unquote, urlsplit, urlunsplit


def clone_url_from_commit_url(commit_url: str, repo_name: str) -> str:
    """Derive a public clone URL from a MegaVul commit URL."""
    if commit_url:
        parsed = urlsplit(commit_url)
        path = unquote(parsed.path).rstrip("/")

        if parsed.netloc.endswith("googlesource.com") and "/+/" in path:
            path = path.split("/+/", 1)[0]
            return urlunsplit(
                (parsed.scheme, parsed.netloc, path, "", "")
            )

        for marker in ("/-/commit/", "/commit/", "/commits/"):
            if marker in path:
                path = path.split(marker, 1)[0]
                break

        if parsed.scheme and parsed.netloc and path:
            if not path.endswith(".git"):
                path += ".git"
            return urlunsplit(
                (parsed.scheme, parsed.netloc, path, "", "")
            )

    if repo_name.count("/") == 1:
        return f"https://github.com/{repo_name}.git"

    raise ValueError(
        f"Cannot derive clone URL for repository {repo_name!r}"
    )

def _stable_id(*parts: str, length: int = 20) -> str:
    raw = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def build_paired_manifest(megavul_path: str | Path) -> list[dict]:
    """Create one real-fix and one synthetic-inverse sample per Java fixing commit.

    MegaVul is function-level and therefore contains repeated commit metadata.
    Only rows marked ``is_vul`` identify functions changed by the vulnerability fix.
    Those rows are grouped back to commit-level observations.
    """
    payload = read_json(megavul_path)
    if not isinstance(payload, list):
        raise ValueError("Expected flattened MegaVul JSON to contain a top-level list")

    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"file_paths": set(), "cve_ids": set(), "git_urls": set(), "parents": set()}
    )
    for row in payload:
        file_path = str(row.get("file_path") or "")
        if not file_path.lower().endswith(".java") or not bool(row.get("is_vul")):
            continue
        project = str(row.get("repo_name") or "").strip()
        commit = str(row.get("commit_hash") or "").strip()
        parent = str(row.get("parent_commit_hash") or "").strip()
        if not project or not commit or not parent:
            continue
        item = grouped[(project, commit)]
        item["file_paths"].add(file_path)
        if row.get("cve_id"):
            item["cve_ids"].add(str(row["cve_id"]))
        if row.get("git_url"):
            item["git_urls"].add(str(row["git_url"]))
        item["parents"].add(parent)

    samples: list[dict] = []
    for (project, commit), item in sorted(grouped.items()):
        if len(item["parents"]) != 1:
            raise ValueError(f"Commit {project}@{commit} has ambiguous parents: {sorted(item['parents'])}")
        parent = next(iter(item["parents"]))
        git_url = sorted(item["git_urls"])[0] if item["git_urls"] else ""
        clone_url = clone_url_from_commit_url(git_url, project)
        pair_id = _stable_id(project, commit)
        common = {
            "pair_id": pair_id,
            "project": project,
            "clone_url": clone_url,
            "commit_hash": commit,
            "parent_commit_hash": parent,
            "cve_ids": sorted(item["cve_ids"]),
            "file_paths": sorted(item["file_paths"]),
        }
        samples.append(
            {
                **common,
                "sample_id": f"fix_{pair_id}",
                "source_revision": parent,
                "target_revision": commit,
                "outcome": FIXING,
                "provenance": OBSERVED_FIX,
            }
        )
        samples.append(
            {
                **common,
                "sample_id": f"inv_{pair_id}",
                "source_revision": commit,
                "target_revision": parent,
                "outcome": INTRODUCING,
                "provenance": SYNTHETIC_INVERSE,
            }
        )

    if not samples:
        raise ValueError("No Java vulnerability-fixing samples were found")
    return samples


def write_paired_manifest(megavul_path: str | Path, output_path: str | Path) -> list[dict]:
    samples = build_paired_manifest(megavul_path)
    write_jsonl(output_path, samples)
    return samples

