from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .constants import METADATA_COLUMNS, OUTCOME
from .io import read_json, read_jsonl, write_json


def feature_name(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    return f"CT__{normalized}"


def parse_change_frequency(path: str | Path, representation: str = "presence") -> dict[str, int]:
    payload = read_json(path)
    counts: Counter[str] = Counter()
    for entry in payload.get("frequencyParent", []):
        raw = str(entry["c"])
        counts[feature_name(raw)] += int(entry["f"])
    if representation == "presence":
        return {name: int(count > 0) for name, count in counts.items()}
    if representation == "count":
        return dict(counts)
    if representation == "binned":
        return {name: min(count, 2) for name, count in counts.items()}
    raise ValueError("representation must be one of: presence, count, binned")


def build_dataset(
    manifest_path: str | Path,
    coming_output_dir: str | Path,
    output_csv: str | Path,
    metadata_json: str | Path,
    representation: str = "presence",
    min_support: int = 1,
) -> pd.DataFrame:
    samples = read_jsonl(manifest_path)
    coming_root = Path(coming_output_dir)
    raw_rows: list[dict] = []
    exclusions: list[dict] = []
    support: Counter[str] = Counter()

    for sample in samples:
        frequency_file = coming_root / sample["sample_id"] / "change_frequency.json"
        if not frequency_file.is_file():
            exclusions.append({"sample_id": sample["sample_id"], "reason": "missing Coming output"})
            continue
        features = parse_change_frequency(frequency_file, representation=representation)
        support.update(features.keys())
        raw_rows.append(
            {
                "sample_id": sample["sample_id"],
                "pair_id": sample["pair_id"],
                "project": sample["project"],
                "commit_hash": sample["commit_hash"],
                "parent_commit_hash": sample["parent_commit_hash"],
                "cve_ids": ";".join(sample["cve_ids"]),
                "provenance": sample["provenance"],
                OUTCOME: sample["outcome"],
                **features,
            }
        )

    # A paired study must never retain just one direction. If either Coming run
    # failed, exclude both members to preserve balance and prevent selection bias.
    pair_members: dict[str, list[dict]] = {}
    for row in raw_rows:
        pair_members.setdefault(row["pair_id"], []).append(row)
    complete_pair_ids = {
        pair_id
        for pair_id, members in pair_members.items()
        if len(members) == 2 and {member[OUTCOME] for member in members} == {"FIXING", "INTRODUCING"}
    }
    for pair_id, members in pair_members.items():
        if pair_id not in complete_pair_ids:
            exclusions.extend(
                {"sample_id": member["sample_id"], "reason": "incomplete paired observation"}
                for member in members
            )
    raw_rows = [row for row in raw_rows if row["pair_id"] in complete_pair_ids]
    support = Counter()
    for row in raw_rows:
        support.update(column for column in row if column.startswith("CT__") and row[column] != 0)
    kept_features = sorted(name for name, value in support.items() if value >= min_support)
    rows = []
    for raw in raw_rows:
        rows.append({**{key: raw[key] for key in METADATA_COLUMNS}, **{f: raw.get(f, 0) for f in kept_features}})
    frame = pd.DataFrame(rows, columns=METADATA_COLUMNS + kept_features)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    write_json(
        metadata_json,
        {
            "representation": representation,
            "min_support": min_support,
            "feature_columns": kept_features,
            "metadata_columns": METADATA_COLUMNS,
            "samples": len(frame),
            "pairs": int(frame["pair_id"].nunique()) if not frame.empty else 0,
            "exclusions": exclusions,
        },
    )
    return frame
